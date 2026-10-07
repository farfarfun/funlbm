# CHANGELOG

本文件记录 funlbm 的版本变更，按时间倒序排列。

## [未发布]

### 修复

- 修复 `server/submit.py` 提交任务时把任务目录路径、交互输入的任务名直接拼接进 shell 命令（`cp`/`sbatch`/`g++`/`nohup`）导致的命令注入与路径含空格失效风险：路径统一用 `shlex.quote` 转义，任务名改为白名单校验（仅允许字母、数字、下划线、短横线），非法任务名在拼接前直接抛 `ValueError`。
- 修复 `funlbm submit --config <非默认文件名>`（如 `job.json`）时，后台 `funlbm run` 仍固定按默认 `./config.json` 启动、与 README 文档行为不符的问题：现在显式传递 `--config <配置文件名>`。
- 修复 `BoundaryConfig`（`input`/`output`/`back` 三个面）在全部使用默认参数构造时因缺少 `or {}` 兜底而抛出 `TypeError` 的问题。
- 修复 `funlbm.lbm.base.Config` 在 `file`/`flow`/`particles` 均使用默认参数构造时因缺少 `or {}`/`or []` 兜底而抛出 `TypeError` 的问题。
- 修复 `ParticleSwarm.__init__` / `create_particle_swarm` 使用可变默认参数 `configs: List[...] = []` 的隐患，改为 `None` + `or []`。
- 修复 `funlbm.server.run` 中配置文件不存在时错误地抛出 `FileExistsError`（应为 `FileNotFoundError`）的问题。
- 修复 `funlbm.server.run` 忽略 `--config` 参数、始终按默认路径加载配置的问题。
- 将 `server/submit.py`、`server/update.py` 中已不存在的 `funbuild.shell.run_shell` 迁移为 `funshell.run_shell`，修复 `funlbm.server` 模块及 `funlbm` 命令行入口完全无法导入的问题。
- 移除 `example/poiseuille/job.py` 中失效的 `Solver` 引用（该类已不存在于代码库中），改为使用当前的 `create_lbm` API。

### 安全

- 移除 `example/poiseuille/job.py` 注释中硬编码的 Aliyun 私有 PyPI 源账号密码（已在公开仓库中暴露约一年）。**该凭据仍需仓库所有者尽快到 Aliyun 控制台吊销/轮换**，代码清理不能撤销已公开的历史泄露。

### 变更

- `pyproject.toml`：`farcache` 依赖下限由 `>=0.1.0` 提升为 `>=2.0.3`，与组织规范（SPEC.md §2）保持一致。
- `pyproject.toml`：显式声明 `license = "MIT"`；依赖下限提升为 `farlog>=1.1.7`；移除仅用于已废弃 `funbuild.shell` 入口的 `funbuild` 运行时依赖，改为 `funshell>=1.0.23`。
- 类型标注从 `typing.Optional/List/Dict/Union` 迁移为 Python 3.10 内置泛型写法（`X | None`、`list[...]`、`dict[...]`）。
- 日志输出统一改用 `farlog`，移除示例函数中的 `print()` 调用。
- 补充公开函数/方法的类型标注与中文 docstring（`funlbm.server`、`funlbm.lbm`、`funlbm.flow`、`funlbm.util`、`funlbm.file`、`funlbm.base`、`funlbm.particle` 等模块）。
- 仓库 GitHub topics 由 9 个精简为 7 个，去掉与 `lattice-boltzmann-method` 重复的泛化标签 `3d`/`lbm`，符合 SPEC.md §11 的 5-8 个数量要求。
- `tests/test_smoke.py`：不再仅记录已知 bug 并跳过测试，替换为真实覆盖 `funlbm.server` 导入、CLI `--help`、以及极小网格（6x6x6）端到端单步模拟的回归测试。

### 构建

- 不提交 `uv.lock`；使用 uv 管理本地开发环境，避免将本机解析结果作为仓库状态。
- README 补充一句话简介、安装命令、最小可运行示例、CLI 用法，并附加组织统一的"关于 farfarfun"区块。

## [1.2.88] 及更早版本

早期版本无 CHANGELOG 记录，具体变更请参考 Git 提交历史。
