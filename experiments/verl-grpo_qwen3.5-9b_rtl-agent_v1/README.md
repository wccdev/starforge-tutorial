# verl Agent Loop RTL 示例：让模型带着编译器的报错改自己的 Verilog

用 verl 官方 Agent Loop 训一个 RTL 设计 agent：模型写一版 SystemVerilog，调
`compile_rtl` / `lint_rtl` 看报错，改一版再试；最终那版拿**三段式奖励**打分。

与 `verl-grpo_qwen3.5-9b_qa-tools_v1` 是同一套官方机制（异步 Agent Loop +
`@function_tool` + 自定义奖励），只换了工具与奖励。那份 README 的坑列表这里全都适用，
下面只写 RTL 特有的部分。

## 为什么奖励是三段而不是一个通过率

只判「testbench 过没过」的奖励训不动：早期 rollout 基本全是 0，组内奖励恒等 →
GRPO 优势全为 0 → **训练照跑但什么都学不到**（与 qa-tools 坑 4 是同一种失败，
病因不同）。三段给的是一条能往上爬的坡：

| 段 | 权重 | 判什么 | 工具 |
| --- | --- | --- | --- |
| `syntax` | 0.1 | 编译得过 —— 入场券 | `iverilog` |
| `simulation` | 0.7 | testbench 过多少，**给部分分** | `iverilog` + `vvp` |
| `synthesis` | 0.2 | 综合得出来，且没推断出锁存器 / 组合环 | `yosys` |

**部分分是核心。** VerilogEval / RTLLM 的 testbench 打印 `Mismatches: 3 in 100 samples`
—— 那是 **0.97** 不是 0。二值化会把「差一点」和「完全不对」抹成同一个数，而它们之间
的距离正是模型要学的东西。

**综合段单独存在的理由**：`yosys` 把「推断出锁存器」报成 warning 后照样退出 0，而
推断出的锁存器在 testbench 里看不出来、在芯片上是个 bug。只看退出码等于放它过去。
`lint_rtl` 工具就是给模型留的抓手 —— 不给它，那 0.2 只能靠运气。

判分实现在 `common/rewards/rtl_reward.py`（与其它奖励一样放共享目录，随作业包上传）；
本目录 `reward.py` 只是 verl 的薄适配器。

## 为什么没有 run_testbench 工具

两个原因，都是硬的：

1. **判卷标准不能交到 agent 手上。** 奖励用的是这道题的隐藏 testbench；给它一个能跑
   testbench 的工具，它学到的就是「怎么试出判据」而不是「怎么设计电路」。
   VerilogEval 的公开/隐藏划分就是这个道理。
2. **`compile_rtl` 与 `lint_rtl` 都不执行任何东西**（`iverilog` 只编译、
   `verilator --lint-only` 只分析）。`vvp` 才真的跑，而 **Verilog 有 `$system`** ——
   跑模型写的 testbench 等价于跑模型写的 shell。执行只发生在奖励那一侧：作业容器里、
   临时目录里、子进程 env 只留 `PATH`、每段带超时。

> ⚠️ 同理：**不要在本地开发机上**直接拿不可信样本跑 `rtl_reward`。容器才是隔离边界。

## 与平台评测的关系

这个奖励的数**不是**平台评测的 pass@1，也不该是：

| | 口径 | 要什么 |
| --- | --- | --- |
| 训练奖励（本实验） | 三段 + 部分分 + 扣综合问题 | **坡度** |
| 平台评测（`sf bench run --suites verilogeval-v2`） | 上游 harness 自带的 testbench 与 pass@k | **可比** |

报告效果时以评测侧为准。训练奖励涨了而评测不涨，通常说明奖励被钻了空子，先看
`Breakdown.to_dict()` 卡在哪一段。

## 镜像要装什么

`iverilog` / `verilator` / `yosys` **都不在 verl 官方镜像里**。缺了的表现：

| 缺哪个 | 表现 |
| --- | --- |
| `iverilog` | 奖励**构造期**报错 → 作业启动即失败（故意的，见下） |
| `yosys` | 同上 |
| `verilator` | 作业能跑，但 `lint_rtl` 每次都回「工具不可用」，模型拿不到综合段的抓手 |

奖励在构造期就报错、而不是运行时静默跳过某一段，是刻意的：**一个悄悄少一段的奖励会让
整轮训练在比声明更小的尺度上跑，而那件事从 reward 曲线上看不出来。** 真要少跑一段，
把对应权重显式设成 0。

自建镜像（`sf` 的自定义镜像流程见平台文档「自定义镜像」）：

```dockerfile
FROM <verl-0.9.0 基础镜像>
RUN apt-get update && apt-get install -y --no-install-recommends \
        iverilog verilator yosys \
    && rm -rf /var/lib/apt/lists/*
```

## 文件说明

| 文件 | 作用 |
| --- | --- |
| `config.yaml` | Agent Loop + 工具路径 + 自定义奖励路径（模型/数据不写在这里） |
| `tools.py` | `compile_rtl` / `lint_rtl` 两个 `@function_tool`，都只做静态分析 |
| `reward.py` | verl `compute_score` 薄适配器 → `common/rewards/rtl_reward.py` |
| `prepare_data.py` | 题库 jsonl → verl parquet（testbench 进 `ground_truth`，不进 prompt） |
| `recipe.lock.json` | 锁 verl 0.9.0 |

## 跑起来

```bash
# 1. 数据（本地一次性）：jsonl 题库 → verl parquet，再推成平台数据集版本
python experiments/verl-grpo_qwen3.5-9b_rtl-agent_v1/prepare_data.py \
    --data-dir datasets/rtl_rl --out-dir /tmp/rtl_parquet
sf dataset push <owner>/rtl-rl /tmp/rtl_parquet

# 2. 校验 + 提交
sf validate experiments/verl-grpo_qwen3.5-9b_rtl-agent_v1
sf submit experiments/verl-grpo_qwen3.5-9b_rtl-agent_v1 \
    --model Qwen/Qwen3.5-9B-Base \
    --train-dataset <owner>/rtl-rl@v1 --train-data train.parquet \
    --validation-dataset <owner>/rtl-rl@v1 --validation-data val.parquet \
    --profile h200:1
```

### 题库从哪来

**不要把 VerilogEval / RTLLM 的题目复制进来当训练集** —— 它们是评测集，训练用了就是
泄题，之后 `sf bench` 的分数不再说明任何事情（平台的数据集互查会把这类重叠标出来）。
自建题库，或用它们的**训练划分**；评测留给 harness。

每行 jsonl：

```json
{"spec": "设计一个 4 位同步计数器，带同步复位……", "testbench": "module tb; … endmodule", "top": "counter"}
```

`testbench` 是**隐藏**的：它进 `reward_model.ground_truth`，不进 prompt。
有一条用例专门守这件事（`test_the_prompt_never_carries_the_hidden_testbench`）。

## RTL 特有的坑

1. **`max_response_length` 按 QA 的经验给** → Verilog 模块本身就长，再算上「生成 →
   报错回灌 → 重写」三轮会被截断。截断的轨迹拿不到完整的 ```verilog 块 → 奖励判
   `FORMAT_PENALTY`，**看起来像模型不听话，其实是预算不够**。本实验给 3072；
   调大 `max_assistant_turns` 时要一起加。
2. **题库里有题没 testbench** → 那道题的 `simulation` 段（0.7）无从算起，整条奖励
   退化成「编译得过就给 0.1」。`prepare_data.py` 在建数据集时直接报错，不让它混进去。
3. **模型把代码包在 ``` 里传给工具** → 原样丢给 `iverilog` 必然编不过，而那不是它
   设计能力的问题。`tools.py` 会剥掉围栏，别让格式噪声吃掉一次工具调用。
4. **模型改错时先复述旧代码再给新版本** → 抠代码取**最后**一个含 `module` 的围栏块。
5. **设计单独编得过、加上 testbench 编不过** → 通常是端口名或位宽和题面对不上。
   奖励把这种情况单独写进 `detail`，别和语法错误混在一起看。
6. **本地 `pytest` 跳过了几条** → `test_compile_rtl_really_compiles` 需要本机装
   `iverilog`（`brew install icarus-verilog`）。没装也不影响其余用例。

## 本地守护测试

```bash
pytest tests/test_rtl_reward.py tests/test_verl_rtl_agent_example.py
```

不需要 GPU、不需要 verl（用 stub 替代 `@function_tool`）。守的是三件事：奖励的坡度与
分段可见性、工具只做静态分析、config 与数据 schema 的契约。
