import os

import click

from funlbm.lbm import LBMD3, Config

document_url = "https://darkchat.yuque.com/org-wiki-darkchat-gfaase/ul41go"


@click.group()
def funlbm():
    pass


@funlbm.command()
@click.option("--config", default="./config.json", help=f"参数配置，{document_url}")
def run(config: str = "./config.json"):
    if not os.path.exists(config):
        info = f"""配置文件不存在，访问{document_url}去配置参数吧"""
        print(info)
        raise FileExistsError(config)
    _config = Config().from_file(config)
    lbm = LBMD3(config=_config)
    lbm.run()
