import os

from funlbm.util import logger

from .base import funlbm_cli

document_url = "https://darkchat.yuque.com/org-wiki-darkchat-gfaase/ul41go"


@funlbm_cli.command()
def run(config: str = "./config.json") -> None:
    """运行 LBM 模拟任务

    根据配置文件创建 LBM 求解器并启动模拟。

    Args:
        config: 配置文件路径，默认为当前目录下的 config.json

    Raises:
        FileNotFoundError: 配置文件不存在时抛出
    """
    from funlbm.lbm import create_lbm

    if not os.path.exists(config):
        info = f"配置文件不存在：{config}，访问 {document_url} 去配置参数吧"
        logger.error(info)
        raise FileNotFoundError(info)
    lbm = create_lbm(config)
    lbm.run()
