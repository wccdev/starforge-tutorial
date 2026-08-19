# experiments/ — 实验目录

每个子目录是一个独立实验（= 一次可提交的作业），命名见 `docs/naming-convention.md`。
探索性调参与交付级方向都放这里——原先的 `projects/` 已并入本目录，不再区分两套布局。

新建：

```bash
cp -r templates/experiment-template experiments/grpo_qwen3.5-4b_gsm8k_v1
```

或用 CLI（推荐，会同时生成 `recipe.lock.json`）：

```bash
sf new grpo_qwen3.5-4b_gsm8k_v1 --method nemo-rl/grpo
```

## 基本要求

每个实验都要有 `README.md`，记录目标、关键超参、结论、SwanLab 链接。

## 交付级实验的额外要求

需要长期维护、可复现、可交付的方向，在上面基础上还要：

- 固定依赖版本与数据集版本
- 完整 eval 流程与基线对比 → `sf eval <实验名>`
- checkpoint 导出（HF 格式）→ `sf export <实验名>`（可 `--push-repo` 推 Hub）
- 完整提交记录（`sf job ls` 看 commit / config 指纹 / run_id，由服务端记录）

## 与「项目」的关系

实验目录不是项目。**项目**是 `starforge.yaml` 里的 `name`，提交后控制台按
`@<用户名>/<项目名>` 把本仓库所有提交聚合成一个项目。项目名不要求全局唯一——
用户名编入分组键，不同用户的同名项目互不干扰。
