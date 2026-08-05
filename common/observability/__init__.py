"""训练侧可观测性采集：主动把训练指标 + 每卡硬件上报到 NeMo Lab Console。

启用方式（由 console 提交作业时注入环境变量，本地直跑则全程 no-op）：
  NEMOLAB_ENDPOINT / NEMOLAB_RUN_ID / NEMOLAB_TOKEN

入口经 scripts/nemolab_boot.py 包装：先 apply_patch() 给 nemo_rl.utils.logger.Logger
挂上 NeMoLabLogger 后端，再运行原始训练入口，无需改 NeMo-RL 源码。

跑 NeMo-RL 以外的框架（FRAMEWORK=custom）时上面那条补丁链路不生效，改用 report 模块——
同一套传输设施，但不依赖 nemo_rl：

    from common.observability import report
    report.init(hparams={...}); report.log({"loss": x}, step=i); report.finish()
    # HF Trainer / TRL：callbacks=[report.NeMoLabCallback()]
"""

__version__ = "0.1.0"
