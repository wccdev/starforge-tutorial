"""多轮「本地文档 grep 检索」Agent 奖励环境（NeMo-RL 0.6.0）。

定位：与 `qa_env.QARewardEnv`（单轮、无工具）做 A/B 对比的**对照组**。
区别只有一个——这里模型可以**多轮调用 `search` 工具检索集群容器内的本地资料**，再作答；
**最终判分复用同一套 qa 奖励**（客观题规则 / 简答题裁判 LLM），保证两实验唯一变量是「能否检索」。

检索方式：在【集群训练进程】所在容器里，对 `DOCS_DIR`（默认 /data/docs，含子目录）下的
**markdown 文件**做检索，把命中片段回灌给模型。后端由 `DOCS_RETRIEVER` 选择：
  - bm25（默认）：纯 Python 自实现的 BM25 相关度检索（进程内懒建倒排索引并缓存），带排序、抗 OCR 噪声；
  - grep：`grep -rinI -F` 递归精确/分词 OR 召回（命中即返回、无排序）。
两者都零外部依赖、零外部服务、结果可解释（回灌片段带文件名+行号），贴合「在容器里查本地资料」的真实工作流。

协议（写进数据集 prompt，见实验 run.py）：
  - 检索资料： <search>关键词</search>            # 环境对本地 markdown 跑 grep，把命中片段作为 observation 回灌
  - 给出答案： 正常作答，并把关键要点放入 \\boxed{...}（与单轮实验完全一致的答案格式）
  每一轮模型输出要么是一次 <search>，要么是带 \\boxed{} 的最终作答：
    - 含 \\boxed{}            → 视为最终答案，复用 qa 奖励判分并结束（不强制必须先检索）
    - 含 <search>…</search>  → 跑 grep 检索本地文档，返回命中片段，继续下一轮
    - 都没有                 → 提示格式，继续（计一轮）
    - 超过 max_turns 仍无答案 → 判 0 结束

奖励分两层：**最终判分**（qa 奖励，训练/验证共用）+ **检索 reward shaping**（检索即时奖励 / 检索后
答对加成 / 无效工具格式或不作答惩罚，**仅训练期**用于引导模型真的去用工具）。验证环境请用
`make_eval_cfg()` 派生 cfg 单独建一个实例，把所有 shaping 归零——否则 NeMo-RL 的
validation/accuracy（= mean(total_reward)）会把工具行为算成准确率，既虚高又无法与无工具 baseline 对比。
"""
from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from typing import Any, Iterator, Optional, TypedDict

import ray
import torch
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

# 确保 Ray actor 进程里能 import 到本仓库的 common 包
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ============================ 本地文档检索工具（BM25 / grep）============================
# 在集群容器内对本地资料目录检索。默认 BM25（带排序的相关度召回，比 grep「命中即返回」更准、抗 OCR 噪声）；
# 也可切回 grep。全部通过环境变量配置（由中心化服务在集群侧注入到作业）：
#   DOCS_RETRIEVER       检索后端：bm25（默认）| grep。bm25 纯 Python 自实现，零外部依赖、结果可解释。
#   DOCS_DIR             资料根目录（含子目录），默认 /data/docs。目录不存在 → 返回占位提示（不抛异常）。
#   DOCS_GLOB            只搜哪些文件，默认 *.md（只搜 markdown）。
#   DOCS_TOP_K           最多回灌几个命中片段（grep 按文件聚合 / bm25 按 chunk），默认 3。
#   DOCS_CONTEXT_LINES   [grep] 每个命中额外带几行上下文（grep -C），默认 2。
#   DOCS_MAX_CHARS       单次检索回灌进上下文的总字符上限，默认 500（短 seq 多轮防 host RAM OOM）。
#   DOCS_MAX_PER_FILE    [grep] 单个文件最多取几处命中（grep -m），默认 3，避免一个文件刷屏。
#   DOCS_TIMEOUT         [grep] 单次 grep 子进程超时（秒），默认 15。
#   DOCS_OR_FALLBACK     [grep] 整句精确匹配查不到时，是否再做「关键词分词 OR 召回」（默认 1 开；0 关）。
#   DOCS_MAX_TERMS       [grep] OR 回退时最多用几个关键词（防止碎词把所有行都召回），默认 12。
#   DOCS_CHUNK_LINES     [bm25] 检索单元（chunk）大小：超长段落按多少行切窗，默认 12。
#   BM25_K1 / BM25_B     [bm25] BM25 超参（词频饱和 / 文档长度归一化），默认 1.5 / 0.75。
#   BM25_MIN_COVERAGE    [bm25] ★相关度下限：最佳 chunk 至少要覆盖查询多少比例的 IDF 质量，默认 0.25。
#                        低于它就返回「未检索到」而不是硬塞 Top-K。用 IDF 覆盖率而非 BM25 绝对分，
#                        是因为绝对分随查询词数/语料规模漂移，没法定一个跨查询通用的阈值；覆盖率天然在 [0,1]。
#   BM25_REL_CUTOFF      [bm25] 相对截断：分数低于 Top1 该比例的 chunk 直接丢弃，默认 0.3。
#                        避免「第 1 条相关、第 2/3 条纯噪声」还是被一起回灌，挤占 DOCS_MAX_CHARS 预算。
#   DOCS_CLEAN           回灌前是否做 markdown/OCR 降噪（去表格符/标题符/图片链接/分隔线、超长行截断、
#                        去重/去空行、加章节标题）。默认 1 开；设 0 回到原始逐行回灌。
#   DOCS_MAX_LINE_CHARS  [clean] 单行最长字符数（截断 OCR/表格超长乱码行），默认 200；0=不限。
# ⚠️ 检索发生在【集群训练进程】（Ray actor）所在容器里，所以 DOCS_DIR 必须是【容器内】真实存在的路径。
DOCS_RETRIEVER = os.environ.get("DOCS_RETRIEVER", "bm25").lower()
DOCS_DIR = os.environ.get("DOCS_DIR", "/data/docs")
DOCS_GLOB = os.environ.get("DOCS_GLOB", "*.md")
DOCS_TOP_K = int(os.environ.get("DOCS_TOP_K", "3"))
DOCS_CONTEXT_LINES = int(os.environ.get("DOCS_CONTEXT_LINES", "2"))
DOCS_MAX_CHARS = int(os.environ.get("DOCS_MAX_CHARS", "500"))
DOCS_MAX_PER_FILE = int(os.environ.get("DOCS_MAX_PER_FILE", "3"))
DOCS_TIMEOUT = float(os.environ.get("DOCS_TIMEOUT", "15"))
DOCS_OR_FALLBACK = os.environ.get("DOCS_OR_FALLBACK", "1") not in ("0", "false", "False", "")
DOCS_MAX_TERMS = int(os.environ.get("DOCS_MAX_TERMS", "12"))
DOCS_CHUNK_LINES = int(os.environ.get("DOCS_CHUNK_LINES", "12"))
BM25_K1 = float(os.environ.get("BM25_K1", "1.5"))
BM25_B = float(os.environ.get("BM25_B", "0.75"))
BM25_MIN_COVERAGE = float(os.environ.get("BM25_MIN_COVERAGE", "0.25"))
BM25_REL_CUTOFF = float(os.environ.get("BM25_REL_CUTOFF", "0.3"))
DOCS_CLEAN = os.environ.get("DOCS_CLEAN", "1") not in ("0", "false", "False", "")
DOCS_MAX_LINE_CHARS = int(os.environ.get("DOCS_MAX_LINE_CHARS", "200"))

# 关键词分词用：英文/数字/缩写/型号（如 CMP、PVD、Qwen3.5）直接抽；中文按 2-gram 滑窗（无需 jieba 也能召回）。
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+#/-]+")
_ZH_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
# 极简中文停用字：跳过含这些字的 2-gram，避免「的X」「在X」之类虚词碎片刷屏。
_ZH_STOP = set("的了和与及或在是有为對对把被让从向到这那此其之也都很更最就还按如并且则等")


def _tokenize(query: str) -> list[str]:
    """把查询切成关键词（OR 召回用），零依赖、不引 jieba：
    - 英文/数字/缩写/型号：正则直接抽（高信息量，原样当关键词）。
    - 中文：≤4 字整体作一个词（精度更好）；更长的按 2-gram 滑窗切，跳过含停用字的 gram。
    返回去重后、最多 DOCS_MAX_TERMS 个关键词（保持出现顺序）。
    """
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = t.strip()
        if len(t) >= 2 and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)

    for tok in _ASCII_TOKEN_RE.findall(query):
        _add(tok)
    for run in _ZH_RUN_RE.findall(query):
        if len(run) <= 4:
            _add(run)
        else:
            for i in range(len(run) - 1):
                bg = run[i:i + 2]
                if bg[0] in _ZH_STOP or bg[1] in _ZH_STOP:
                    continue
                _add(bg)
    return terms[:DOCS_MAX_TERMS]


def _iter_terms(text: str) -> Iterator[str]:
    """逐个产出关键词（**不去重、不截断**，全小写）——BM25 建索引数词频(tf)用。
    分词规则与 _tokenize 完全一致（英文/型号正则；中文 ≤4 字整体、更长按 2-gram 跳停用字），
    区别只是这里要保留重复出现以统计词频，且不限制数量。
    """
    for tok in _ASCII_TOKEN_RE.findall(text):
        if len(tok) >= 2:
            yield tok.lower()
    for run in _ZH_RUN_RE.findall(text):
        if len(run) <= 4:
            if len(run) >= 2:
                yield run.lower()
        else:
            for i in range(len(run) - 1):
                bg = run[i:i + 2]
                if bg[0] in _ZH_STOP or bg[1] in _ZH_STOP:
                    continue
                yield bg.lower()


# ============================ 回灌降噪 / 预算感知拼装（省 token、去噪声）============================
# 把 markdown/OCR 原始行清洗成「干净正文」再回灌：去掉对模型无信息但占 token 的符号噪声，
# 并对超长乱码行做截断。清洗只影响【回灌文本与 BM25 词频统计】，不影响行号（空行用 "" 占位保号）。
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")            # ![alt](url) → 整体删
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")           # [text](url) → 保留 text
_MD_RULE_RE = re.compile(r"^\s*([-=*_])\1{2,}\s*$")          # --- / === / *** 分隔线
_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$")  # # 标题
_HTML_TAG_RE = re.compile(r"</?[A-Za-z][^>]*>")              # 裸 HTML 标签
_WS_RE = re.compile(r"[ \t\u3000]+")                          # 折叠空白（含全角空格）


def _heading_text(line: str) -> Optional[str]:
    """是 markdown 标题行就返回标题文字，否则 None。"""
    m = _MD_HEADING_RE.match(line)
    return m.group(2).strip() if m else None


def _clean_md_line(line: str) -> str:
    """markdown/OCR 行降噪。返回清洗后的正文；纯噪声（分隔线/空壳）返回 ""（调用方按空行处理）。

    DOCS_CLEAN=0 时只做最基本的右侧去空白，保持原样。
    """
    if not DOCS_CLEAN:
        return line.rstrip()
    s = line.strip()
    if not s or _MD_RULE_RE.match(s):
        return ""
    s = _MD_IMAGE_RE.sub("", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = _HTML_TAG_RE.sub("", s)
    # 表格行 | a | b | c | → a  b  c（去掉竖线与对齐分隔行）
    if s.startswith("|") or " | " in s:
        cells = [c.strip() for c in s.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):  # |---|:--:| 这种对齐行整行丢
            return ""
        s = "  ".join(c for c in cells if c)
    # 去行首 markdown 记号：# 标题号 / > 引用 / 列表符 / 序号
    s = re.sub(r"^\s{0,3}(#{1,6}\s+|>\s?|[-*+]\s+|\d+[.)]\s+)", "", s)
    s = _WS_RE.sub(" ", s).strip()
    if DOCS_MAX_LINE_CHARS and len(s) > DOCS_MAX_LINE_CHARS:
        s = s[:DOCS_MAX_LINE_CHARS].rstrip() + "…"
    return s


def _assemble_blocks(blocks: list[str]) -> str:
    """把若干片段块拼进 DOCS_MAX_CHARS 预算：尽量多塞几块，最后一块按【行边界】截断，绝不从行中间切。

    比旧的 "\\n---\\n".join(blocks)[:N] 更好：① 不会把一块从中间切碎；② 预算在 TopK 间分配，
    避免第一块过长把后面的块整体挤掉。
    """
    if not blocks:
        return ""
    sep = "\n---\n"
    out: list[str] = []
    used = 0
    for b in blocks:
        add_len = len(b) + (len(sep) if out else 0)
        if used + add_len <= DOCS_MAX_CHARS:
            out.append(b)
            used += add_len
            continue
        # 预算不够整块：在行边界放下能放的部分（剩余空间够放至少一行才放）
        remaining = DOCS_MAX_CHARS - used - (len(sep) if out else 0)
        if remaining > 48:
            head = b[:remaining]
            cut = head.rfind("\n")
            if cut > 0:
                out.append(head[:cut] + "\n  ⋯（截断）")
        break
    return sep.join(out)


_LINENO_PREFIX_RE = re.compile(r"^L\d+:\s*")


def _dedup_keep_order(lines: list[str], seen: set[str]) -> list[str]:
    """跨块去重：丢掉正文已出现过的实质性行（去重键剥掉「Lxx: 」行号前缀，故不同位置的相同内容也能去重）。
    仅对长度≥6 的内容去重，避免误删短编号/标题行；空行/省略号原样保留。"""
    kept: list[str] = []
    for ln in lines:
        key = _LINENO_PREFIX_RE.sub("", ln.strip())
        if len(key) >= 6:
            if key in seen:
                continue
            seen.add(key)
        kept.append(ln)
    return kept


def _run_grep(terms: list[str]) -> tuple[int, str, str]:
    """对 DOCS_DIR 下的 markdown 跑一次 grep；多个 term 用多个 -e（固定字符串、OR 语义、无需转义正则）。
    返回 (returncode, stdout, stderr)；returncode<0 表示子进程异常（超时/缺 grep）。
    """
    cmd = [
        "grep", "-rinI", "-F",
        f"-C{max(0, DOCS_CONTEXT_LINES)}",
        f"-m{max(1, DOCS_MAX_PER_FILE)}",
        f"--include={DOCS_GLOB}",
    ]
    for t in terms:
        cmd += ["-e", t]   # 每个 -e 是一个固定字符串模式，命中任一即算（OR）
    cmd += ["--", DOCS_DIR]  # -- 终止选项解析，防止 term/路径以 '-' 开头被当成参数
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOCS_TIMEOUT)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return -1, "", type(e).__name__
    return proc.returncode, proc.stdout, proc.stderr


def _grep_search(query: str) -> str:
    """对本地 markdown 文档跑 grep，返回拼好的命中片段文本（失败/未命中返回提示，不抛异常）。

    两段式：先整句精确匹配（高精度）；查不到再把查询分词后做 OR 召回（高召回，DOCS_OR_FALLBACK 开关）。
    （调用方 docs_search 已做 query 规整与目录存在性检查。）
    """
    # 第一段：整句精确匹配（固定字符串）。
    rc, out, err = _run_grep([query])
    if rc == 0:
        return _format_grep_output(out, [query])
    if rc < 0:
        return f"search 错误: 检索失败（{err}）"
    if rc > 1:
        msg = (err or "").strip().splitlines()
        return f"search 错误: grep 返回 {rc}{('：' + msg[0]) if msg else ''}"

    # 第二段：整句没命中（rc==1），分词后 OR 召回。
    if DOCS_OR_FALLBACK:
        terms = _tokenize(query)
        # 仅当分词结果跟整句不同（即确实拆出了多个/不同关键词）才值得再查一次。
        if terms and terms != [query]:
            rc2, out2, err2 = _run_grep(terms)
            if rc2 == 0:
                return _format_grep_output(out2, terms)
            if rc2 < 0:
                return f"search 错误: 检索失败（{err2}）"
            if rc2 > 1:
                msg = (err2 or "").strip().splitlines()
                return f"search 错误: grep 返回 {rc2}{('：' + msg[0]) if msg else ''}"

    return "未检索到相关资料（换个关键词再试）"


def _block_file_path(first_line: str) -> Optional[str]:
    """从块首行里切出文件绝对路径。

    grep -r 每行是 `<文件路径><分隔><行号><分隔><内容>`（命中用 ':'，上下文用 '-'）。
    优先用 DOCS_GLOB 的扩展名（如 .md）+ 紧跟的分隔符来定位路径结尾——
    这样即便文件名里含 '-数字-'（如 v1-2-foo.md）也不会切错；扩展名取不到时退回首个 `<分隔><数字><分隔>`。
    """
    ext = DOCS_GLOB.replace("*", "")  # "*.md" -> ".md"
    if ext:
        m = re.search(re.escape(ext) + r"[:-]", first_line)
        if m:
            return first_line[: m.start() + len(ext)]
    m = re.match(r"^(.+?)[:-]\d+[:-]", first_line)
    return m.group(1) if m else None


def _parse_grep_line(line: str, base: str) -> Optional[tuple[str, Optional[int], str]]:
    """解析 grep -r 的一行 → (相对路径, 行号或None, 正文)。无法解析返回 None。

    每行形如 `<文件路径>:<行号>:<命中行>`（命中）或 `<文件路径>-<行号>-<上下文行>`（上下文）。
    路径定位复用 _block_file_path（按扩展名，文件名含 '-数字-' 也不会切错）。
    """
    absfile = _block_file_path(line)
    if not absfile:
        return None
    rel = absfile[len(base):] if absfile.startswith(base) else absfile
    rest = line[len(absfile):] if line.startswith(absfile) else line
    mm = re.match(r"^[:-](\d+)[:-]?(.*)$", rest)
    if mm:
        return rel, int(mm.group(1)), mm.group(2)
    return rel, None, rest


def _format_grep_output(raw: str, terms: list[str]) -> str:
    """把 grep -r 的原始输出**按文件聚合**成片段，按命中关键词数排序后取前 TOP_K，再按字符上限截断。

    排序：命中块多于 DOCS_TOP_K 时，**优先保留命中关键词更多的文件块**
    （按该文件正文命中的不同 term 数降序；同分保持 grep 原始（即文件首次出现）顺序）。
    term 命中只在【正文】里数，不含文件路径前缀，避免文件名误计。
    按文件分组而非按 '--' 分块：grep 在 -C0 时不输出 '--'，分组才对 context=0 也健壮，也正好对应「文件块」。
    """
    base = DOCS_DIR.rstrip("/") + "/"
    lowered_terms = [t.lower() for t in terms if t]

    files: dict[str, list[tuple[Optional[int], str]]] = {}
    order: list[str] = []
    for line in raw.splitlines():
        if not line.strip() or line == "--":
            continue
        parsed = _parse_grep_line(line, base)
        if not parsed:
            continue
        rel, lno, text = parsed
        if rel not in files:
            files[rel] = []
            order.append(rel)
        files[rel].append((lno, text))

    if not order:
        return "未检索到相关资料（换个关键词再试）"

    scored: list[tuple[int, int, str]] = []  # (命中 term 数, 文件首次出现序号, 排版后文本)
    seen: set[str] = set()
    for idx, rel in enumerate(order):
        rows = files[rel]
        content = "\n".join(t for _, t in rows).lower()
        score = sum(1 for t in lowered_terms if t in content)
        body: list[str] = []
        prev: Optional[int] = None
        for lno, text in rows:
            cleaned = _clean_md_line(text)
            if not cleaned:  # 清洗后为纯噪声/空 → 跳过（不占 token）
                continue
            if prev is not None and lno is not None and lno - prev > 1:
                body.append("  ⋯")  # 同文件内不连续的命中区域之间插省略号
            body.append(f"L{lno}: {cleaned}" if lno is not None else cleaned)
            prev = lno
        body = _dedup_keep_order(body, seen)
        if not any(b.strip() and b.strip() != "⋯" for b in body):
            continue
        scored.append((score, idx, f"【{rel}】\n" + "\n".join(body)))

    if not scored:
        return "未检索到相关资料（换个关键词再试）"
    scored.sort(key=lambda x: (-x[0], x[1]))  # 命中多的文件优先；同分稳定（保持首次出现顺序）
    out_blocks = [b for _, _, b in scored[:DOCS_TOP_K]]
    return _assemble_blocks(out_blocks)


# ============================ BM25 检索（纯 Python，零依赖）============================
# grep 是「命中即返回」、无相关度排序，OCR 噪声文档下召回质量差（模型据此学到「检索没用」→ 放弃检索）。
# BM25 给每个 chunk 算相关度分、取 Top-K，召回与排序都更稳，且仍是本地、零外部服务、结果可解释（带文件名+行号）。
# 索引在 actor 进程内**懒构建一次并缓存**（训练期资料不变）；分词复用上面零依赖的 _iter_terms（含中文 2-gram）。
class _Bm25Index:
    """一个资料目录的 BM25 倒排索引。chunk = (相对路径, 起始行号, 章节标题, 该 chunk 清洗后的行列表)。"""

    __slots__ = ("chunks", "postings", "idf", "doc_len", "avgdl", "n")

    def __init__(self) -> None:
        self.chunks: list[tuple[str, int, str, list[str]]] = []
        self.postings: dict[str, list[tuple[int, int]]] = {}  # term -> [(chunk_id, tf), ...]
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avgdl: float = 1.0
        self.n: int = 0


# 进程内缓存：DOCS_DIR -> 索引（None 占位表示「资料库为空」，避免反复重建）。
_BM25_CACHE: dict[str, Optional[_Bm25Index]] = {}


def _iter_doc_files() -> Iterator[str]:
    """遍历 DOCS_DIR（含子目录）下匹配 DOCS_GLOB 后缀的文件。"""
    suffix = DOCS_GLOB.replace("*", "")  # "*.md" -> ".md"
    for root, _dirs, files in os.walk(DOCS_DIR):
        for fn in files:
            if not suffix or fn.endswith(suffix):
                yield os.path.join(root, fn)


def _split_chunks(path: str) -> list[tuple[str, int, str, list[str]]]:
    """把一个文件切成检索单元：按空行分段；段落超过 DOCS_CHUNK_LINES 行再按窗口切。

    每个 chunk 附带它所属的最近 markdown 章节标题（给模型定位用），并在切窗时存【清洗后】的行
    （清洗后的纯噪声行用 "" 占位以保持行号正确，输出时再过滤）。
    """
    try:
        raw = open(path, encoding="utf-8", errors="ignore").read()
    except OSError:
        return []
    lines = raw.splitlines()
    base = DOCS_DIR.rstrip("/") + "/"
    rel = path[len(base):] if path.startswith(base) else path

    out: list[tuple[str, int, str, list[str]]] = []

    def _emit(start_lno: int, buf: list[str], heading: str) -> None:
        cleaned = [_clean_md_line(s) for s in buf]
        if not any(s for s in cleaned):
            return
        for off in range(0, len(cleaned), DOCS_CHUNK_LINES):
            window = cleaned[off:off + DOCS_CHUNK_LINES]
            if any(s for s in window):
                out.append((rel, start_lno + off, heading, window))

    para: list[str] = []
    para_start = 1
    current_heading = ""
    for i, ln in enumerate(lines, start=1):
        h = _heading_text(ln)
        if h is not None:
            current_heading = h
        if ln.strip() == "":
            _emit(para_start, para, current_heading)
            para = []
            para_start = i + 1
        else:
            if not para:
                para_start = i
            para.append(ln)
    _emit(para_start, para, current_heading)
    return out


def _build_bm25_index(docs_dir: str) -> Optional[_Bm25Index]:
    """遍历资料目录，切 chunk、分词、建倒排与 IDF。资料为空返回 None。"""
    idx = _Bm25Index()
    for f in _iter_doc_files():
        idx.chunks.extend(_split_chunks(f))
    idx.n = len(idx.chunks)
    if idx.n == 0:
        return None

    df: dict[str, int] = {}
    idx.doc_len = [0] * idx.n
    for cid, (_rel, _start, _heading, lines) in enumerate(idx.chunks):
        tf: dict[str, int] = {}
        for term in _iter_terms(" ".join(lines)):
            tf[term] = tf.get(term, 0) + 1
        idx.doc_len[cid] = sum(tf.values()) or 1
        for term, c in tf.items():
            idx.postings.setdefault(term, []).append((cid, c))
            df[term] = df.get(term, 0) + 1

    idx.avgdl = sum(idx.doc_len) / idx.n
    # BM25 标准 IDF（带 +1 平滑，恒非负）。
    idx.idf = {t: math.log(1 + (idx.n - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    return idx


def _bm25_search(query: str) -> str:
    """BM25 召回 Top-K chunk，拼成「【文件】Lxx: 内容」片段（与 grep 输出风格一致）。

    ★ 带相关度下限：只有「跟查询确实相关」的 chunk 才回灌。判据是 **IDF 覆盖率**——
    最佳 chunk 命中的查询词 IDF 之和 / 查询全部词的 IDF 之和 ≥ BM25_MIN_COVERAGE。
    分母含语料里根本不存在的词（按索引最大 IDF 计），所以「关键术语查不到、只蹭到几个常见词」
    会被正确判为不相关。达不到下限就返回「未检索到」，让上层 _is_useful_retrieval 判为无效检索，
    既不发检索奖励、也不置 did_search——否则模型「搜到一堆无关资料」也照样拿分。
    """
    if DOCS_DIR not in _BM25_CACHE:
        _BM25_CACHE[DOCS_DIR] = _build_bm25_index(DOCS_DIR)
    idx = _BM25_CACHE[DOCS_DIR]
    if idx is None:
        return "未检索到相关资料（资料库为空）"

    q_terms = set(_iter_terms(query))  # query 一般短，每个 term 计一次贡献即可
    if not q_terms:
        return "未检索到相关资料（换个关键词再试）"
    # 语料里不存在的查询词按「最大 IDF」计入分母：它们是查不到的关键术语，应当拉低覆盖率。
    max_idf = max(idx.idf.values()) if idx.idf else 1.0
    total_idf = sum(idx.idf.get(t, max_idf) for t in q_terms) or 1.0

    scores: dict[int, float] = {}
    covered: dict[int, float] = {}  # chunk -> 命中的查询词 IDF 之和
    for term in q_terms:
        post = idx.postings.get(term)
        if not post:
            continue
        w = idx.idf[term]
        for cid, tf in post:
            dl = idx.doc_len[cid]
            denom = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / idx.avgdl)
            scores[cid] = scores.get(cid, 0.0) + w * (tf * (BM25_K1 + 1)) / denom
            covered[cid] = covered.get(cid, 0.0) + w
    if not scores:
        return "未检索到相关资料（换个关键词再试）"

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best_cid, best_score = ranked[0]
    if covered.get(best_cid, 0.0) / total_idf < BM25_MIN_COVERAGE:
        # 最相关的一条都没覆盖到足够的查询信息量 → 整次检索判为未命中，宁可返回空也不返回噪声。
        return "未检索到相关资料（相关度过低，换个更具体的关键词）"
    # 相对截断：只保留跟 Top1 同一量级的 chunk，避免噪声块挤占 DOCS_MAX_CHARS 预算。
    floor = best_score * BM25_REL_CUTOFF
    top = [(cid, sc) for cid, sc in ranked[:DOCS_TOP_K] if sc >= floor]

    blocks: list[str] = []
    seen: set[str] = set()
    for cid, _score in top:
        rel, start, heading, lines = idx.chunks[cid]
        rows = [f"L{start + j}: {ln}" for j, ln in enumerate(lines) if ln.strip()]
        rows = _dedup_keep_order(rows, seen)
        if not rows:
            continue
        title = f"【{rel} ▸ {heading}】" if heading else f"【{rel}】"
        blocks.append(title + "\n" + "\n".join(rows))
    if not blocks:
        return "未检索到相关资料（换个关键词再试）"
    return _assemble_blocks(blocks)


# ============================ 检索分派入口 ============================
def docs_search(query: str) -> str:
    """本地资料检索入口：按 DOCS_RETRIEVER 选 BM25（默认）或 grep。失败/未命中返回提示，不抛异常。

    换检索方式（向量检索等），只在此分派即可，环境其余逻辑不变。
    """
    query = " ".join((query or "").split())  # 折叠空白/去换行
    if not query:
        return "search 错误: 查询为空"
    if not os.path.isdir(DOCS_DIR):
        return f"（本地资料目录未接入：DOCS_DIR={DOCS_DIR} 不存在或不可访问。请联系管理员确认容器内已挂载资料。）"
    if DOCS_RETRIEVER == "grep":
        return _grep_search(query)
    return _bm25_search(query)


# ============================ 元数据 / 文本解析 ============================
class QADocsMetadata(TypedDict, total=False):
    expected_answer: str   # 带 [type] 前缀的金标准（与单轮实验一致）
    query: str             # 题面（裁判 LLM / 检索上下文用）
    num_turns: int         # 已交互轮数
    max_turns: int         # 最大轮数
    did_search: bool       # 轨迹中是否真正取回过资料（reward shaping：答对加成用）


def _extract_tag(text: str, tag: str) -> Optional[str]:
    """取最后一个 <tag>...</tag> 的内容；没有开标签则 None。

    闭标签缺失时仍取开标签后全文：NeMo-RL / vLLM 用 stop_strings=["</search>"] 截断时，
    默认常不把 </search> 写进生成文本，若这里要求成对标签会误判「格式不对」，
    白白烧掉一轮（max_turns=2 时几乎等于检索失败）。
    """
    open_t, close_t = f"<{tag}>", f"</{tag}>"
    s = text.rfind(open_t)
    if s == -1:
        return None
    e = text.find(close_t, s + len(open_t))
    body = text[s + len(open_t) : (e if e != -1 else None)].strip()
    return body if body else ("" if e != -1 else None)


def _last_assistant_text(message_log: LLMMessageLogType) -> str:
    for msg in reversed(message_log):
        if msg.get("role") == "assistant":
            return str(msg.get("content", "")).strip()
    return ""


# docs_search 在「失败/未命中/目录未接入」时返回的提示都以这些前缀开头；
# 用它判断一次检索是不是真的取回了资料（只有真取到才给检索奖励 / 记 did_search）。
_SEARCH_FAIL_PREFIXES = ("search 错误", "未检索到相关资料", "（本地资料目录未接入")


def _is_useful_retrieval(obs: str) -> bool:
    """这次检索是否真的取回了资料片段（非错误、非未命中、非目录未接入）。"""
    s = (obs or "").lstrip()
    return bool(s) and not s.startswith(_SEARCH_FAIL_PREFIXES)


# ==================== 训练 / 验证的 reward shaping 分离（验证必须用 make_eval_cfg）====================
# 下面这几项是【训练期的探索引导】，不属于「答得对不对」：search_step_reward / answer_search_bonus
# 会把分数抬到 1.0 以上，no_answer_penalty 会压到 0 以下。
# 而 NeMo-RL 的 validation/accuracy 直接就是 mean(total_reward)，total_reward 又是**逐轮奖励的累加**
# （见 nemo_rl/experience/rollouts.py），所以验证若照抄训练 cfg：
#   ① 「用了工具」本身就白送分（max_turns=3 时最多 +0.05×2 检索 + 0.1 加成 = +0.2 绝对值），验证分虚高；
#   ② 与单轮无工具 baseline（qa-rl_v1，纯最终判分）不再同尺度，A/B 结论会被工具加分污染。
# 故验证环境一律用 make_eval_cfg() 派生的 cfg。
_SHAPING_KEYS = (
    "search_step_reward",
    "answer_search_bonus",
    "no_answer_penalty",
    "no_search_answer_penalty",
    "format_error_penalty",
    "invalid_search_penalty",
)


def make_eval_cfg(cfg: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """由训练 env cfg 派生【验证 / 评测用】cfg：reward shaping 全部归零，只保留最终答案判分。

    检索后端与判分方式（use_judge / max_turns 等）与训练完全一致——验证与训练的唯一差异是
    「用不用工具不再额外加分或扣分」：答对就是 1.0，答错 0.0，超轮不作答也是 0.0。
    于是 validation/accuracy 就是纯粹的答题得分，可直接与无工具 baseline 对比。
    """
    out = dict(cfg or {})
    out.update(dict.fromkeys(_SHAPING_KEYS, 0.0))
    # 判分口径（short_answer_scope）刻意**不**归零：它是「答得对不对」的定义，训练和验证必须一致。
    # 只把统计打上 val 标签，好在作业日志里把两个 actor 的 [qa_docs_stats] 行区分开——
    # 验证期的 search_rate 才是这个实验真正的过程指标（训练期的会被 shaping 抬高）。
    out["stats_tag"] = "val"
    return out


# ============================ 环境 ============================
@ray.remote  # pragma: no cover
class QADocsAgentEnv(EnvironmentInterface[QADocsMetadata]):
    """多轮本地文档 grep 检索 QA 环境（Ray Actor）。最终判分复用 common/rewards 的 qa 奖励。"""

    SEARCH_STOP_STRINGS = ["</search>"]

    def __init__(self, cfg: Optional[dict[str, Any]] = None):
        global DOCS_MAX_CHARS
        self.cfg = cfg or {}
        self.use_judge = bool(self.cfg.get("use_judge", True))
        retrieval_max_chars = self.cfg.get("retrieval_max_chars")
        if retrieval_max_chars is not None:
            DOCS_MAX_CHARS = int(retrieval_max_chars)
        # ── 检索 reward shaping（鼓励模型真的去用工具，而不是退化成闭卷瞎猜）──
        # 观测到的问题：奖励只看最终 \boxed 对错，对「检索动作」零回报，且 grep 偶有噪声，
        # 于是 RL 把策略收敛到「不检索、直接答常识题」→ 准确率早早卡在 ~62%、专有知识题系统性全错。
        # 这里给「真正取回资料的检索」一点即时奖励 + 「检索后答对」一次性加成，并惩罚「只检索不作答」防刷分。
        #   search_step_reward    每次「有效检索」（真取回片段）的即时奖励。小于答对收益，仅作探索引导。
        #   answer_search_bonus   最终答对(≥min)且轨迹检索过的一次性加成（奖励"靠检索答对"）。
        #   search_bonus_min_score  触发上面 bonus 的最低 base 分。
        #     ⚠️ 别设 1.0：multiple 走 partial_penalty、fill 是「答对空数/总空数」、short 是关键词覆盖率，
        #        这三类几乎永远拿不到正好 1.0，而它们恰恰是【最依赖检索的专有知识题】；
        #        设 1.0 等于只给 single/bool（模型闭卷就能蒙）发检索加成，激励方向正好反了。
        #   search_bonus_scaled   true=加成按得分比例给（bonus×r，梯度更平滑，推荐）；false=达到阈值就给满额。
        #   no_answer_penalty     超 max_turns 仍无 \boxed 的惩罚（让"光检索不答"净收益为负，防 reward hacking）。
        #     ⚠️ 这条要生效，env 的 max_turns 必须 <= grpo.max_rollout_turns - 1，详见 step() 里的说明。
        # ⚠️ 这几项只该作用在【训练】：它们是探索引导，不是答题正确性。
        #    验证/评测请用 make_eval_cfg(cfg) 另建一个环境实例（shaping 全零），否则 validation/accuracy
        #    会把「用了工具」的加分算进去而虚高，也无法与无工具 baseline 同尺度对比。
        self.search_step_reward = float(self.cfg.get("search_step_reward", 0.05))
        self.answer_search_bonus = float(self.cfg.get("answer_search_bonus", 0.1))
        self.search_bonus_min_score = float(self.cfg.get("search_bonus_min_score", 0.5))
        self.search_bonus_scaled = bool(self.cfg.get("search_bonus_scaled", True))
        self.no_answer_penalty = float(self.cfg.get("no_answer_penalty", 0.2))
        # 闭卷直接 \boxed 作答的惩罚：防止「检索 0 回报 + 选择题可蒙对」把策略推向不用工具。
        # 默认 0 保持旧行为；agent 实验可显式设 >0。验证由 make_eval_cfg 归零。
        self.no_search_answer_penalty = float(self.cfg.get("no_search_answer_penalty", 0.0))
        # NeMo-RL v0.7 的 invalid_tool_call_advantage 只支持 NeMo-Gym。
        # 本实验走原生 EnvironmentInterface，因此在环境奖励层处理无标签输出和空 <search>，
        # 并由 make_eval_cfg() 在验证时归零，避免污染 accuracy。
        self.format_error_penalty = float(self.cfg.get("format_error_penalty", 0.0))
        self.invalid_search_penalty = float(self.cfg.get("invalid_search_penalty", 0.0))
        # short 题「关键词覆盖率」统计范围："boxed" 只认答案框，"completion" 连整段回答一起算。
        # ⚠️ 检索 Agent 应当显式设 "boxed"：模型手里有回灌的资料原文，若整段回答都算覆盖，
        #    它只要复述检索片段就能刷满 short 分——单轮 baseline 没有这条通道，
        #    而 make_eval_cfg() 归零的是 reward shaping、管不到判分口径，验证分会一起虚高。
        # 默认保持 "completion"（= qa_reward 的历史行为），避免静默改变其他实验的判分口径；
        # 各实验在自己的 config 里 opt-in。
        import common.rewards.qa_reward as _qa_reward

        _qa_reward.SHORT_SCOPE = str(self.cfg.get("short_answer_scope", "completion"))
        # ── 检索行为统计（诊断用，不参与奖励）──────────────────────────────────
        # 为什么需要：本环境最危险的失败模式是「模型学会不检索」——validation/accuracy 照样能涨
        # （涨的是闭卷答题水平），但实验的核心变量已经没了。而 EnvironmentInterface 的
        # global_post_process_and_metrics 在 NeMo-RL 0.7 里【从未被调用】（rollouts.py 只调 step.remote），
        # 往那里加指标不会有任何输出。所以这里自己累计 + 落盘 + 打日志。
        self._stats_tag = str(self.cfg.get("stats_tag", "train"))
        self._stats_path = self.cfg.get("stats_path") or os.environ.get("QA_DOCS_STATS_PATH")
        self._stats_print_every = int(self.cfg.get("stats_print_every", 20))
        self._reset_stats()
        # 与 QARewardEnv 同源：客观题走规则；简答 use_judge=true 走裁判、失败回退关键词覆盖率。
        if self.use_judge:
            from common.rewards.qa_judge_reward import qa_judge_reward_fn

            self._reward_fn = qa_judge_reward_fn
        else:
            from common.rewards.qa_reward import qa_rule_reward_fn

            self._reward_fn = qa_rule_reward_fn
        # boxed 检测复用 qa_reward 的实现（正确处理嵌套花括号）
        from common.rewards.qa_reward import extract_boxed

        self._extract_boxed = extract_boxed

    # ── 检索行为统计 ─────────────────────────────────────────────────────────
    def _reset_stats(self) -> None:
        self._stats = {
            "step_calls": 0,        # step() 被调用次数（≈ rollout 轮数 × 批次数）
            "search_attempts": 0,   # 模型发起 <search> 的次数
            "useful_retrievals": 0, # 其中真取回相关资料的次数（过了 BM25 相关度下限）
            "answers": 0,           # 给出 \boxed{} 最终答案的轨迹数
            "answers_with_search": 0,   # 其中检索过的轨迹数 ← 这个比例就是「检索率」
            "no_answer_penalized": 0,   # 超轮不作答被罚的轨迹数（恒为 0 说明惩罚分支不可达！）
            "format_errors": 0,
        }

    def _flush_stats(self) -> None:
        """把累计统计打到 stdout（落进作业日志，可 grep `[qa_docs_stats]`），可选再落一行 JSONL。

        指标含义：
          search_rate            = answers_with_search / answers  ← **最关键**：掉到 0 就是塌缩成闭卷了
          useful_retrieval_rate  = useful_retrievals / search_attempts ← 检索质量（BM25 调参看这个）
          no_answer_penalized    恒为 0 且 search_attempts 很高 → 惩罚分支不可达，检查 max_turns 配置
        """
        s = self._stats
        ans, att = max(1, s["answers"]), max(1, s["search_attempts"])
        line = (
            f"[qa_docs_stats] tag={self._stats_tag} "
            f"search_rate={s['answers_with_search'] / ans:.3f} "
            f"useful_retrieval_rate={s['useful_retrievals'] / att:.3f} "
            f"answers={s['answers']} search_attempts={s['search_attempts']} "
            f"no_answer_penalized={s['no_answer_penalized']} "
            f"format_errors={s['format_errors']}"
        )
        print(line, flush=True)
        if self._stats_path:
            try:
                import json as _json

                with open(self._stats_path, "a", encoding="utf-8") as f:
                    f.write(_json.dumps({"tag": self._stats_tag, **s}, ensure_ascii=False) + "\n")
            except OSError:
                pass  # 统计落盘失败绝不能影响训练

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[QADocsMetadata],
    ) -> EnvironmentReturn[QADocsMetadata]:
        n = len(message_log_batch)
        observations: list[dict[str, str]] = [None] * n  # type: ignore[list-item]
        rewards: list[float] = [0.0] * n
        terminateds: list[bool] = [False] * n
        next_stops: list[Optional[list[str]]] = [None] * n
        next_meta: list[Optional[QADocsMetadata]] = [None] * n
        answers: list[Optional[list[str]]] = [None] * n

        # 收集"本轮给出最终答案"的样本，最后批量判分（简答裁判是并发批处理，批量更省）
        final_idx: list[int] = []
        final_q: list[str] = []
        final_comp: list[str] = []
        final_exp: list[str] = []
        final_searched: list[bool] = []  # 该样本轨迹中是否真正取回过资料（用于答对加成）

        for i, (log, meta) in enumerate(zip(message_log_batch, metadata, strict=True)):
            content = _last_assistant_text(log)
            num_turns = int(meta.get("num_turns", 0))
            max_turns = int(meta.get("max_turns", 4))
            expected = str(meta.get("expected_answer", ""))
            query = str(meta.get("query", ""))
            did_search = bool(meta.get("did_search", False))  # 跨轮累积：之前是否有效检索过

            boxed = self._extract_boxed(content)
            search_q = _extract_tag(content, "search")

            # 1) 最终答案（含 \boxed{}）：批量判分后结束。不强制必须先检索。
            if boxed is not None:
                final_idx.append(i)
                final_q.append(query)
                final_comp.append(content)
                final_exp.append(expected)
                final_searched.append(did_search)
                terminateds[i] = True
                answers[i] = [expected]
                continue

            # 2) 超过最大轮数仍无答案：判负（惩罚「只检索不作答」式刷分）结束。
            # ⚠️ 可达性：NeMo-RL 的 rollout 是 `for turn in range(max_rollout_turns)`
            #    （nemo_rl/experience/rollouts.py），循环跑完就直接退出、**不会再调一次 step()**，
            #    只把样本标记为 max_turns_reached（rollouts.py 末尾），既不补判分也不扣分；
            #    而 grpo 的 overlong_filtering 只对 truncated 置零 loss，管不到这种轨迹。
            #    num_turns 每轮 +1，所以第 k 轮进来时 num_turns == k-1：
            #    要让本分支触发，必须 max_turns <= max_rollout_turns - 1。
            #    若两者相等（曾经的配置），本分支永远不可达 →「搜满 N 次不作答」净收益是
            #    +N×search_step_reward 的**正数**，成为零风险刷分策略。
            #    统计里的 no_answer_penalized 恒为 0 就是这个症状。
            if num_turns >= max_turns:
                rewards[i] = -self.no_answer_penalty
                self._stats["no_answer_penalized"] += 1
                observations[i] = {"role": "environment", "content": f"已达最大轮数 {max_turns}，结束。"}
                terminateds[i] = True
                continue

            nm: QADocsMetadata = dict(meta)  # type: ignore[assignment]
            nm["num_turns"] = num_turns + 1
            nm["did_search"] = did_search

            # 3) 检索本地文档：grep 返回片段，继续。真取回资料才给即时检索奖励并记 did_search。
            if search_q is not None:
                obs = docs_search(search_q)
                self._stats["search_attempts"] += 1
                if _is_useful_retrieval(obs):
                    rewards[i] = self.search_step_reward
                    nm["did_search"] = True
                    self._stats["useful_retrievals"] += 1
                elif not search_q:
                    rewards[i] = -self.invalid_search_penalty
                observations[i] = {"role": "environment", "content": f"[检索结果]\n{obs}"}
                next_stops[i] = self.SEARCH_STOP_STRINGS
                next_meta[i] = nm
                continue

            # 4) 格式不对：提示并重试（计一轮）
            rewards[i] = -self.format_error_penalty
            self._stats["format_errors"] += 1
            observations[i] = {
                "role": "environment",
                "content": (
                    "格式不对。检索本地资料用 <search>关键词</search>；"
                    "作答把关键要点放入 \\boxed{...}（多个用 ; 分隔）。"
                ),
            }
            next_stops[i] = self.SEARCH_STOP_STRINGS
            next_meta[i] = nm

        # 批量判分给出最终答案的样本；「检索后答对」额外加成，「未检索就作答」训练期扣分。
        if final_idx:
            scores = self._reward_fn(final_q, final_comp, final_exp)
            for i, s, searched in zip(final_idx, scores, final_searched, strict=True):
                r = float(s)
                self._stats["answers"] += 1
                if searched:
                    self._stats["answers_with_search"] += 1
                # 加成按得分比例给（search_bonus_scaled，默认开）：fill/short/multiple 拿的是
                # 0~1 的连续分，"达阈值就给满额"会在阈值处制造一个断崖，梯度不连续；
                # 乘以 r 则「答得越好、靠检索拿到的加成越多」，方向一致且平滑。
                if searched and r >= self.search_bonus_min_score:
                    bonus = self.answer_search_bonus * (r if self.search_bonus_scaled else 1.0)
                else:
                    bonus = 0.0
                skip_pen = 0.0 if searched else self.no_search_answer_penalty
                rewards[i] = r + bonus - skip_pen
                tags: list[str] = []
                if bonus:
                    tags.append(f"+检索加成 {bonus:.3f}")
                if skip_pen:
                    tags.append(f"-未检索 {skip_pen:.3f}")
                tag = f"  ({'; '.join(tags)})" if tags else ""
                observations[i] = {"role": "environment", "content": f"得分: {r:.3f}{tag}"}

        self._stats["step_calls"] += 1
        if self._stats_print_every > 0 and self._stats["step_calls"] % self._stats_print_every == 0:
            self._flush_stats()

        return EnvironmentReturn(
            observations=observations,
            metadata=next_meta,
            next_stop_strings=next_stops,
            rewards=torch.tensor(rewards, dtype=torch.float32),
            terminateds=torch.tensor(terminateds, dtype=torch.bool),
            answers=answers,
        )

    def shutdown(self):
        pass

    def global_post_process_and_metrics(
        self, batch: BatchedDataDict
    ) -> tuple[BatchedDataDict, dict]:
        # ⚠️ NeMo-RL 0.7 从不调用这个钩子（rollouts.py 只调 step.remote），下面的指标不会出现在任何
        #    日志里。检索行为请看 _flush_stats() 打到作业日志的 `[qa_docs_stats]` 行。
        #    保留本方法只为满足 EnvironmentInterface；若将来框架接上了钩子，这里会自动生效。
        rewards = batch.get(
            "total_reward", torch.tensor([0.0] * len(batch["idx"]))
        ).float()
        if len(rewards) == 0:
            return batch, {}
        metrics = {
            "qa_docs_mean_reward": rewards.mean().item(),
            "qa_docs_perfect_rate": (rewards >= 1.0).float().mean().item(),
            "qa_docs_format_penalty_rate": (rewards < 0).float().mean().item(),
        }
        return batch, metrics
