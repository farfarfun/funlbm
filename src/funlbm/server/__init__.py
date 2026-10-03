# 注意：此处必须用 `from . import run, submit, update` 而不是
# `from .run import run`/`from .submit import submit`/`from .update import update`。
# 后者会把包属性 `funlbm.server.run`/`submit`/`update` 从子模块对象覆盖成同名函数，
# 导致 `import funlbm.server.submit as m` 拿到的是函数而不是模块（已在排查中发现，
# 并曾导致 tests/test_smoke.py 里对应的 monkeypatch 测试报错）。
from . import run, submit, update  # noqa: F401  (导入即触发 Typer 子命令注册)
from .base import funlbm

__all__ = ["funlbm"]
