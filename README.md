# funlbm

基于 PyTorch 的三维格子玻尔兹曼方法（LBM）数值模拟库，支持流固耦合的颗粒仿真、
多种离散速度模型（D3Q13/D3Q15/D3Q19/D3Q27）、断点续算与 HDF5/VTK/Tecplot 数据导出。

## 安装

```bash
pip install funlbm
```

## 最小可运行示例

准备一份 `config.json`（网格大小、边界条件、颗粒参数等），然后：

```python
from funlbm.lbm import create_lbm

lbm = create_lbm("./config.json")
lbm.run(max_steps=1000)
```

更完整的配置示例见仓库内 [`example/`](example) 目录（`freefall`、`poiseuille` 等场景）。

## 命令行用法

安装后会提供 `funlbm` 命令行工具：

```bash
funlbm run --config ./config.json      # 前台运行一次模拟
funlbm submit --config ./config.json   # 提交任务（本地/HPC 算力平台）
funlbm update                          # 更新到最新版本
```

## 相关链接

[pylbm](https://github.com/pylbm/pylbm)

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
