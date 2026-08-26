# common/ — 跨实验复用代码

随作业包上传。只放**你的**数据脚本、环境、奖励、算法插件，不要放平台采集。

- `data/`         — 数据集下载 / 清洗 / 转换
- `environments/` — 自定义 Environment（GRPO 奖励来源；多轮 Agent）
- `rewards/`      — 规则奖励 / LLM 裁判
- `algorithms/`   — 示例算法插件（OPSD / MaxRL）
- `envkit/`       — 工具调用与 gym 适配
- `eval/`         — 评测脚本共用逻辑
- `bootstrap.py`  — NeMo-RL 实验 run.py 样板

指标和硬件曲线来自平台包，训练脚本或环境里写：

```python
from starforge.report import init, log, finish
```

`observability: platform` 的 custom 作业会把 `starforge` 放进 `PYTHONPATH`。
NeMo-RL / verl / TRL 由 runner 自动接框架 logger，不必再拷一份采集库。
