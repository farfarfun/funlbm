from farlog import getLogger
from funshell import run_shell

from .base import funlbm_cli

logger = getLogger("funlbm")


@funlbm_cli.command()
def update() -> None:
    """更新 funlbm 包。"""
    logger.info("开始更新")
    run_shell("pip install funlbm -U")
    logger.success("更新成功")
