# RTL 内部数据怎么收集

面向"用公司内部 RTL 数据训一个设计 agent"。先说结论:

> **收集一次 `spec + reference + testbench` 三件套,SFT / GRPO / DPO / OPSD 的数据全都能从它派生出来。**

原因是 testbench 让这份数据成为一个**验证器**,而不只是一批样本。有验证器就能自动生成偏好对、自动打二值标签、自动筛难度;没有验证器,那些都得靠人标。

---

## 1. 收集格式

一行一道题的 jsonl:

```json
{
  "id": "sync_fifo_01",
  "spec": "实现模块 sync_fifo,接口如下:\n  module sync_fifo #(parameter W=8, D=16) (\n    input clk, input rst_n,\n    input wr_en, input [W-1:0] din,\n    input rd_en, output reg [W-1:0] dout,\n    output full, output empty);\n同步 FIFO,深度 D,rst_n 为低电平同步复位。full 时忽略写入,empty 时 dout 保持。",
  "top": "sync_fifo",
  "reference": "module sync_fifo #(...)  ... endmodule",
  "testbench": "`timescale 1ns/1ps\nmodule tb; ... endmodule",
  "meta": {"source": "git:a1b2c3d", "tags": ["fifo", "时序"], "difficulty": "medium"}
}
```

| 字段 | 必需 | 谁用它 |
| --- | --- | --- |
| `spec` | ✅ | 所有方法的 prompt |
| `top` | ✅ | 综合段(`yosys -top`);也用来校验 spec 有没有钉死模块名 |
| `testbench` | RL 必需 | GRPO 的判据;派生 DPO/KTO 时的自动标注器 |
| `reference` | 强烈建议 | SFT 的 target、OPSD 的老师输入、**自检脚本的变异基准** |
| `id` / `meta` | 建议 | 追溯、按标签分层、划分训练/验证 |

`prepare_data.py` 只读 `spec` / `testbench` / `top`,`reference` 和 `meta` 它不用——但**别因此不收**,它们是其它三种方法的原料,也是自检的前提。

---

## 2. testbench 的三条硬要求

奖励的 0.7 全压在这里。testbench 有毛病时训练**不会报错**,只会安静地不涨。

### 2.1 必须打印可解析的错误计数

```verilog
$display("Mismatches: %0d in %0d samples", errors, total);
```

格式就是这一句(判分正则 `[Mm]ismatches:\s*(\d+)\s+in\s+(\d+)\s+samples`)。

**没有这一行,奖励退化成二值**——而二值奖励在 GRPO 里基本训不动:一组 8 条 rollout 全是 0,组内优势恒为 0,这道题这一步白跑。有计数才有"差 3 个 = 0.97、差 30 个 = 0.7"的坡度。

> 你们现有的 testbench 大概率是给人看日志的,不是给机器打分的。**主要的数据工程量就在这里:给每个 testbench 补一个错误计数器和这一句汇总。**

### 2.2 必须一定会结束

```verilog
initial begin
  #200000;                                          // 超时兜底,别只靠正常路径
  $display("Mismatches: %0d in %0d samples", errors, total);
  $finish;
end
```

漏了 `$finish`、或者被组合环卡住 → `vvp` 永远不返回 → 超时判 0。而"超时"和"设计错"在 reward 曲线上长得一模一样,归因极难。

### 2.3 自检,不要靠人眼看波形

testbench 自己算期望、自己比对、自己计数。dump VCD 在训练里没有任何意义——没人看。

> ⚠️ 调试用的 `$display("ERROR: ...")` 会被判成**失败标记**。要么删掉,要么确保汇总计数行也在(计数优先级更高)。

---

## 3. spec 必须钉死接口

**最隐蔽的杀手。** spec 没说清端口名/位宽 → 模型写出逻辑完全正确的模块 → testbench 里 `.wr_en(...)` 连不上 → 判"带 testbench 编译失败" → **0.1 封顶,无论设计多好**。

所以 spec 里要把 module header 原样贴出来(见上面例子)。VerilogEval 就是这么做的。必须钉死的:

- 模块名(要和 `top` 一致,和 testbench 里实例化的名字一致)
- 端口名、方向、位宽
- 时钟/复位极性,同步还是异步
- 参数名与默认值

自检脚本会在 spec 里找不到模块名时给出提醒。

---

## 4. 从公司内部哪里挖

按价值排序:

| 来源 | 怎么变成题 | 价值 |
| --- | --- | --- |
| **修 bug 的提交** | 父提交的坏代码 + issue 描述 → spec;修复后跑得过的 testbench → 判据 | **最高** |
| 现有模块 + 它的 testbench | 模块的设计文档/注释 → spec;模块本身 → reference | 高,量最大 |
| 设计规格文档 | 文档一节 → spec,再补 testbench | 中,要新写 testbench |
| 代码评审意见 | "这里会推断出锁存器" → 专门练综合段的题 | 中,量少但对症 |

**第一类最值钱**:它天然带着"坏代码 + 症状 + 判据",而且你们 git 里已经有了。扫历史时优先挑**改动 ≤ 50 行、且同一提交里 testbench 没变**的——testbench 没变说明判据是稳定的,那份 testbench 直接可用。

spec 可以让 LLM 从代码和 commit message 生成初稿,但**接口那一段必须人工过一遍**(理由见第 3 节)。

---

## 5. 难度筛选 —— 最容易被忽略的一步

GRPO 的优势是**同一道题的 n 条 rollout 之间**算的。所以:

- n 条全部编不过 → 全 0.1 → 这道题这一步**白跑**
- n 条全部满分 → 全 1.0 → 同样白跑

**有用的题,是基座模型"有时候能做对"的那些。** 正式训练前先筛一遍:

```
用基座模型对每道题采样 8 次 → 用同一套三段式奖励打分 → 保留奖励标准差 > 0 的题
```

两头都扔掉(或者把难题留一小部分做课程学习的后段)。这一步能省掉一大半 GPU 时间。

> 三段式奖励在这里有额外好处:它比二值奖励更难出现"全同分",因为部分分把题目切细了。同一道题,二值下可能 8 条全 0,三段下会分出 0.1 / 0.38 / 0.8 三档。

---

## 6. 数量

| 阶段 | 量级 | 什么时候需要 |
| --- | --- | --- |
| SFT 热启动 | 几百 ~ 几千对 `(spec, reference)` | 基座**连编都编不过**时。先拿 50 道题试,syntax 段通过率 < 30% 就先 SFT |
| GRPO 训练集 | **筛选后 200~500 道** | 主训练。题少不是问题,每道题跑多轮 epoch |
| 验证集 | 50~100 道 | 独立,不参与训练 |

**质量远比数量重要**:一道 testbench 写得好的题,胜过十道判不出错的题。

---

## 7. 划分与防泄题

- **不要把 VerilogEval / RTLLM / CVDP 的题放进训练集。** 训了之后 `sf bench` 的分数就不再说明任何事情。平台的数据集互查会把这类重叠标出来。
- 验证集和训练集**按模块划分**,不要按行随机切——同一个 FIFO 的两个变体分到两边就是泄题。
- 用平台的版本化数据集:`sf dataset push <owner>/rtl-rl <目录>`,提交时用 `--train-dataset <owner>/rtl-rl@v1` 引用。

---

## 8. 上训练前先自检

```bash
python experiments/verl-grpo_qwen3.5-9b_rtl-agent_v1/check_data.py \
    datasets/rtl_rl/train.jsonl --write-bad /tmp/bad.jsonl
```

需要本机装 `iverilog`(`brew install icarus-verilog` / `apt install iverilog`)。五道检查:

1. 字段齐全,`top` 和 spec 里的模块名对得上
2. 参考实现编得过
3. **参考实现拿满分** —— 拿不到说明 testbench 判错了自己的正确答案
4. **把参考实现改坏,分数必须掉** ← 最关键
5. 能在超时内跑完

第 4 条是这个脚本存在的全部理由。**testbench 判不出错**是最贵的坑:所有题都满分,训练看起来在涨,其实什么都没学到——而且没有任何报错。脚本会做几种最小变异(反转加法、反转相等判断、移位反向……),只要有一个能让分数掉下来,就说明这道题至少能判出一类错误。

---

## 9. 一份数据,四种练法

这是收集 `reference` 和 `testbench` 的回报:同一份题库能派生出各种方法要的格式。

### SFT(冷启动)

NeMo-RL `ResponseDataset` 吃 `{"input", "output"}`:

```python
{"input": row["spec"], "output": f"```verilog\n{row['reference']}\n```"}
```

**什么时候用**:基座模型连合法 Verilog 都写不出来时。RL 需要偶尔的成功来产生梯度,一次成功都没有就先 SFT 把 syntax 段拉起来。

### GRPO / MaxRL(主训练)

就是本实验的格式,`prepare_data.py` 直接产出。**只有这条路能真的把"功能对不对"变成训练信号**——因为只有它把 testbench 跑起来了。

### DPO / 奖励模型(偏好)

偏好对**不用人标,用 testbench 自动生成**:

```
对每道题采样 n 条 → 用三段式奖励打分 → 高分做 chosen、低分做 rejected
```

⚠️ **必须卡阈值**:`chosen` 要真的通过(比如 ≥ 0.9),`rejected` 要真的失败(比如 ≤ 0.3)。不卡的话你会拿 0.15 对 0.10 去配对,那是在教模型"偏好一个稍微没那么坏的错误答案"——比不训还糟。中间地带的样本直接丢掉。

**什么时候用**:已经有 RL 模型、想再稳一档;或者手上有大量人工评审意见(评审的"这样写更好"天然就是偏好对)。

### KTO(二值偏好)

比 DPO 更省:不需要配对,每条样本一个二值标签。

```
采样 → 奖励 ≥ 阈值 记 True,否则 False
```

**什么时候用**:采样出来的分布很偏(比如九成都失败),配不出足够多的对。

### OPSD(同策自蒸馏)

本仓库 `opsd_qwen3.5-9b_math_h200_1n2g` 用的格式是 `{"problem", "solution"}`:

```python
{"problem": row["spec"], "solution": row["reference"]}
```

⚠️ OPSD 的老师 = 同一个模型 + **额外看到参考解**。所以 `solution` 必须是**完整的参考实现**,不能只给一句"答案是 XXX"——老师看不到实现过程就退化成和学生一样只看题目,KL 恒等于 0,整个训练白跑。这也是为什么 `reference` 值得收。

**什么时候用**:想把大模型的能力压到小模型上,或者在没有可靠 testbench 的题目上也想训(OPSD 不需要验证器)。

### 一张表

| 方法 | 需要的字段 | 需要 testbench? | 典型用途 |
| --- | --- | --- | --- |
| SFT | `spec` + `reference` | ❌ | 冷启动,让它先写出合法 Verilog |
| GRPO / MaxRL | `spec` + `testbench` + `top` | ✅ | 主训练,把"功能对不对"变成信号 |
| DPO / RM | `spec` + `testbench`(自动配对) | ✅ | 精修;或利用人工评审意见 |
| KTO | `spec` + `testbench`(自动打标) | ✅ | 采样分布很偏时的替代 |
| OPSD / 蒸馏 | `spec` + `reference` | ❌ | 压到小模型;无验证器的题也能训 |

---

## 10. 建议的推进顺序

1. **先收 30~50 道**,把 testbench 按第 2 节改造好,跑 `check_data.py` 跑通全绿。这一步会暴露你们 testbench 写法上的系统性问题,早发现比收了 500 道再返工便宜得多。
2. 用基座模型跑一次,看 syntax 段通过率——决定要不要先 SFT。
3. 扩到 200~500 道,做第 5 节的难度筛选。
4. GRPO 训练,用 `sf bench run --suites verilogeval-v2` 做**独立**的效果评估(训练奖励涨不代表评测涨)。
