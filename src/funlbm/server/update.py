from funbuild.shell import run_shell
from funutil import getLogger
from .base import funlbm_cli

logger = getLogger("funlbm")


@funlbm_cli.command()
def update():
    logger.info("开始更新")
    run_shell("pip install funlbm -U")
    logger.success("更新成功")
