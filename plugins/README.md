# 插件开发指南

插件 = 一个带 `plugin.yaml` 的目录，发布到平台后可被任何人安装、锁定进实验、随作业注入。
本目录既是官方插件包的源码位置，也是教程：

| 目录 | 类型 | 说明 |
| --- | --- | --- |
| `examples/rloo/` | algorithm（eager） | **教程样板**：自包含的优势估计器插件，从零写算法插件先抄它 |
| `examples/tiny-qa-prep/` | data-prep | **教程样板**：数据预处理插件，生成合成 QA 数据 |
| `maxrl/` | algorithm（eager） | 官方插件：`common/algorithms/maxrl` 的适配层 |
| `opsd/` | algorithm（deferred） | 官方插件：deferred（需 tokenizer 上下文）的参考实现 |

## 一个插件长什么样

```
my-plugin/
├── plugin.yaml        # manifest（必须，放目录根）
└── my_module.py       # 你的代码（algorithm 类型才需要）
```

`plugin.yaml` 全部字段：

```yaml
schema: lab-plugin/v1
name: my-plugin           # 叶子名（字母数字开头，仅含字母/数字/. _ -）；
                          # 对外 ID = <owner>/<name>，owner 由平台在发布时按你的账号盖章
version: 0.1.0            # 不可变：改了内容必须换版本号，同版本重发会被 409 拒绝
kind: algorithm           # algorithm | data-prep
load: eager               # 仅 algorithm：eager | deferred（见下）
summary: 一句话说明        # 显示在 Web 插件中心与 sf plugin ls
entrypoint: my_module:install   # 仅 algorithm：模块路径相对包根
requires:
  sdk: ">=2.1,<3"         # PEP 440；装载前校验，不满足直接拒跑
```

## 类型一：algorithm（算法补丁，跑在训练容器里）

作业提交时平台把插件包注入到作业目录 `forge_plugins/<name>/`，launcher 校验
digest 后按 manifest 装载。入口签名固定：

```python
def install(params, **ctx) -> None: ...
```

- `params`：该作业的 hyperparams 快照（dict）。
- `ctx`：eager 恒为空；deferred 由训练入口传（如 `pad_token_id`、`max_seq_len`）。

**eager**（绝大多数情况）：launcher 在训练进程启动前直接调用 `install`。适合
monkey-patch 框架行为 —— 完整可跑的样板见 [`examples/rloo/`](examples/rloo/)，
套路是给框架的工厂函数包一层、新增自己的分支、幂等防重复安装。

**deferred**（需要运行时上下文才能装载时才用）：launcher 只登记不执行，训练入口
拿到 tokenizer 等上下文后统一经 `common.algorithms.registry.install_deferred(name, **ctx)`
装载（注册表会优先取插件包版本）。参考 [`opsd/`](opsd/)。

注意：算法插件跑在**你自己的训练容器**里，import 得到的是上传作业包 + 集群镜像
里的依赖。插件包自身应自包含（别 import 别人作业包里的模块）；需要新 Python
依赖时确认镜像里有 —— 平台不做插件级依赖安装。

## 类型二：data-prep（数据预处理脚本，跑在你本机）

不需要 entrypoint。安装到仓库 `forge_plugins/<name>/` 后，CLI 按文件名约定
`prepare_<数据集名>.py` 自动发现：

```
sf dataset prepare              # 列出所有可用数据集（内置 + 插件）
sf dataset prepare tiny_qa      # 执行 forge_plugins/*/prepare_tiny_qa.py
```

脚本首行 docstring 作为一句话说明展示在列表里；同名时仓库内置（`common/data/`）
优先，插件不能悄悄替换内置数据集。完整样板见 [`examples/tiny-qa-prep/`](examples/tiny-qa-prep/)。

## 发布 → 安装 → 使用

```bash
# 1. 发布（打包目录上传，平台解包校验 manifest、算内容 digest 盖章）
sf plugin publish plugins/examples/rloo

# 2. 查看货架
sf plugin ls
sf plugin info <owner>/rloo

# 3. 安装：下载到本地 forge_plugins/，并锁定到实验（写 plugins.lock.json）
sf plugin install <owner>/rloo --exp experiments/my-exp

# 4. 正常提交；提交时平台按锁文件校验并注入，launcher 装载前再验一次 digest
sf submit experiments/my-exp
```

digest 是插件目录内容的 sha256（`__pycache__`/`.pyc` 等不参与），在发布、提交、
装载三个环节各校验一次 —— 锁定的是**内容**而不只是版本号，任何一环内容不符都拒绝执行。

发布前本地自检：

```python
from starforge_sdk.plugins import load_manifest, directory_digest
m = load_manifest(Path("plugins/examples/rloo"))   # manifest 不合法这里就会抛
print(m, directory_digest(Path("plugins/examples/rloo")))
```

## 约束速查

- 包大小 ≤ 32 MB（插件是代码，不是数据/权重 —— 那些走 dataset/artifact 通道）
- 版本不可变；管理员可禁用插件（只拦新提交，不影响已在跑的作业）
- 只能发布到自己的命名空间（admin 可用 `--owner` 跨）
- `requires.sdk` 不满足时 launcher 拒跑，写清楚它能省一次白排队
