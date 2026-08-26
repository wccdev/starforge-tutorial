# 大模型微调考试操作手册

> 面向 StarForge 微调平台（格科内网）的实操考试指南。拿到本文档 + HR 发放的账号后，按顺序完成即可。

---

## 一、考试概览

| 项目 | 说明 |
| --- | --- |
| **任务目标** | 使用 **GRPO**，在**技术培训考题**上微调 **Qwen 3.5 9B**，使验证集 **accuracy** 尽可能高 |
| **训练方式** | GRPO + **多轮工具调用**（模型可多次检索 `/data/docs` 中的 markdown 资料后再作答） |
| **模型** | `Qwen/Qwen3.5-9B-Base`（LoRA，单卡 H100 80GB） |
| **数据** | 全员共用 `/data/datasets/qa_rl`（训练 / 验证集，平台已挂载） |
| **GPU / 时限** | 每人 **1 张 GPU**，**48 小时** |
| **评分** | **validation/accuracy**（验证集准确率） |

### 任务要点

1. 模型作答前可**多轮检索**集群内 `/data/docs` 技术资料，再给出答案。
2. 最终答案须写入 `\boxed{...}`（如 `\boxed{B}`、`\boxed{A,C}`），否则格式扣分。
3. 题型含单选 / 多选 / 判断 / 填空 / 简答；简答由平台内置 LLM 裁判打分。
4. **需自行实现完整实验**：包括多轮检索 Agent 环境、`run.py`、`config.yaml` 及调参。仓库仅提供两个示例实验作参考，**不提供现成答案**。

---

## 二、账号与网络

### 2.1 HR 发放的账号

| 账号 | 用途 |
| --- | --- |
| **VPN 账号** | 连接格科内网 |
| **StarForge 控制台账号** | Web 控制台 + CLI 提交作业 |

### 2.2 VPN（深信服 EasyConnect）

微调平台在内网，须先连 VPN。

1. 下载安装 **EasyConnect**（向 HR 索取安装包，或访问内网下载页）。
2. 填入 VPN 服务器地址，用 HR 账号登录。

**张江站点双链路**（卡顿可切换）：

| 链路 | 地址 |
| --- | --- |
| 联通 | `https://zjvpn.gcoreinc.com:11880` |
| 电信 | `https://zjvpn.gcoreinc.com:11800` |

其他办公地点请向 HR 确认 VPN 地址。

连上 VPN 后，浏览器能打开 [https://starforge.gcoreinc.com/](https://starforge.gcoreinc.com/) 即表示网络正常。

### 2.3 Web 控制台

访问 [https://starforge.gcoreinc.com/](https://starforge.gcoreinc.com/)，用 HR 账号登录。主要用 **作业** 页查看训练曲线、验证样本与日志。

---

## 三、Fork 仓库与本机环境

### 3.1 Fork 示例仓库（第一步）

在 GitHub 上 **Fork** 官方示例仓库到你自己的账号：

```
https://github.com/wccdev/starforge-tutorial
```

Fork 后你在自己的仓库里创建实验、改代码、提交 commit。**考试期间的所有改动都在你的 Fork 里完成**，不要直接在官方仓库改。

### 3.2 克隆你的 Fork 并安装 CLI

```bash
pip install starforge
git clone https://github.com/<你的GitHub用户名>/starforge-tutorial.git
cd starforge-tutorial
```

本机只是提交客户端，**不需要 GPU**。NeMo-RL / CUDA 等在远程集群容器内。

```bash
sf ...
```

### 3.3 登录微调平台

**确保 VPN 已连接**：

```bash
sf login --server https://starforge.gcoreinc.com
```

SSH / 无浏览器：`sf login --device-flow`

```bash
sf status    # 确认已登录、服务可达、配额
sf status    # 查看 GPU 配额与活跃作业
```

---

## 四、任务说明

### 4.1 交互协议（需自行实现）

模型与环境的典型交互：

```
<search>关键词</search>
→ 环境在 /data/docs 检索 markdown，回灌 [检索结果]
→ 可多次检索
→ 最终 \boxed{答案}
```

具体协议设计、环境 `step()` 逻辑、奖励计算等**由你自行实现**。

### 4.2 平台资源（已就绪，勿改）

| 资源 | 集群路径 |
| --- | --- |
| 训练 / 验证集 | `/data/datasets/qa_rl` |
| 可检索资料 | `/data/docs` |

数据格式：每行 JSON，`{"query": "...", "expected_answer": "[single] B"}` 等。提交作业时平台自动注入 `QA_RL_DATA_DIR`，**无需在本机准备数据**。

### 4.3 仓库里可参考的内容

| 实验 | 可参考什么 |
| --- | --- |
| `grpo_qwen3.5-9b_gsm8k_v1` | 单轮 GRPO 实验结构 |
| `agent-grpo_qwen3.5-9b_sliding-puzzle_v1` | 多轮 Agent GRPO 怎么跑通 |

其余（自定义环境、数据加载、`run.py`、超参）请结合仓库 `common/`、`configs/` 和两个示例实验**自行研究**。

### 4.4 创建你的实验

```bash
sf ls
sf new grpo_qwen3.5-9b_qa-rl-agent_<你的名字> --from agent-grpo_qwen3.5-9b_sliding-puzzle_v1
```

在你的 Fork 里完成实验代码后，**push 到 GitHub**：

```bash
git add .
git commit -m "add qa-rl agent experiment"
git push origin main
```

提交训练时 CLI 会把当前 working-dir 打包上传集群，请确保 push 前代码已保存。

---

## 五、提交训练与监控

### 5.1 提交

```bash
sf validate <你的实验名>    # 提交前校验 config
sf submit <你的实验名>
```

成功后会打印 **作业 ID**（如 `raysubmit_xxx`）。

```bash
sf job logs [job_id]   # 查看日志
sf job ls              # 作业列表
sf job stop <job_id>   # 停止作业、释放 GPU
```

### 5.2 控制台监控

[https://starforge.gcoreinc.com/](https://starforge.gcoreinc.com/) → **作业** → 点击你的作业：

- **图表**：关注 **validation/accuracy**（主要评分指标）
- **验证样本**：查看模型检索与作答轨迹
- **日志 / 系统 / 诊断**：排查失败与 OOM

48 小时内可多次 submit 调参；及时 `sf job stop` 释放不用的作业。

---

## 六、考试规则与结果提交

### 6.1 规则

- 每人 1 GPU，限时 48 小时（起止时间以 HR 通知为准）
- **不得修改** `/data/datasets/qa_rl` 数据
- **不得** 共用账号或抄袭他人配置
- 以**最佳 validation/accuracy** 为准

### 6.2 截止时间前，按 HR 要求提交

| 提交项 | 说明 |
| --- | --- |
| **GitHub Fork 地址** | 你的 `starforge` Fork 仓库 URL（含实验代码） |
| **作业 ID** | 最佳 run 对应的 Ray 作业 ID |
| **截图** | 控制台 validation/accuracy 曲线或最终数值 |
| **简要说明** | 实验思路、关键改动、最佳 accuracy 等（HR 指定格式为准） |

---

## 七、常见问题

| 问题 | 处理 |
| --- | --- |
| 打不开 starforge | 确认 VPN；切换张江双链路 |
| `sf login` 无浏览器 | `sf login --device-flow` |
| 配额不足 | `sf status`；`sf job stop` 释放卡 |
| 作业 FAILED | 控制台看日志 / 诊断；`sf job logs <id> -n 0` |
| validate 失败 | 按终端报错改 config |

---

## 八、操作流程 Checklist

- [ ] GitHub Fork `wccdev/starforge-tutorial`
- [ ] 克隆自己的 Fork，`pip install starforge`
- [ ] 连 VPN，登录 [starforge.gcoreinc.com](https://starforge.gcoreinc.com/)
- [ ] `sf login` + `sf status`
- [ ] 研究两个示例实验，创建并实现自己的 QA 多轮检索实验
- [ ] 代码 push 到 GitHub Fork
- [ ] `sf validate` → `sf submit`
- [ ] 监控曲线，迭代调参
- [ ] 向 HR 提交：Fork 地址 + 作业 ID + 截图 + 简要说明

---

## 附录：常用命令

```bash
pip install starforge
sf login --server https://starforge.gcoreinc.com
sf status
sf ls
sf new <实验名> --from agent-grpo_qwen3.5-9b_sliding-puzzle_v1
sf validate <实验名>
sf submit <实验名>
sf job logs [job_id]
sf job stop <job_id>
```

---

如有平台或配额问题，在微信群联系 HR 或考试负责人。
