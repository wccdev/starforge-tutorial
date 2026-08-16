# projects/ — 正式 / 交付级项目

放需要长期维护、可复现、可交付的微调项目。布局与 `experiments/` 一致，但要求更高：

- 固定依赖版本与数据集版本
- 完整 eval 流程与基线对比 → `sf eval <项目名>`（封装 NeMo-RL `run_eval.py`）
- checkpoint 导出（HF 格式）流程 → `sf export <项目名>`（自适应 dcp/megatron，可 `--push-repo` 推 Hub）
- 完整实验记录与 SwanLab 链接（提交记录用 `lab runs` 看：commit / config 指纹 / run_id，由服务端记录）

新建同样从模板拷贝：

```bash
cp -r templates/experiment-template projects/<项目名>
```
