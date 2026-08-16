# common/ — 项目共享代码

实验通过 `common.*` 引用的自定义代码放这里，随作业包上传到集群：

| 目录 | 放什么 | 何时需要 |
| --- | --- | --- |
| `common/data/` | 数据预处理脚本 `prepare_*.py`（`sf dataset prepare` 自动发现） | 需要预处理内部数据时 |
| `common/environments/` | 多轮 Agent 环境（工具调用 / 可验证任务） | Agent RL 实验 |
| `common/rewards/` | 自定义奖励函数（关键词 / LLM 裁判） | RL 实验需要自定义奖励时 |

纯 SFT / 数学 GRPO 实验用官方基底即可，不需要写 common 代码。
Agent 环境与裁判/沙箱的写法见平台文档「Agent 环境套件」。
