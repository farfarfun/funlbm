import os

import click

from funlbm.lbm import create_lbm

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
    lbm = create_lbm()
    lbm.run()


cli = typer.Typer(help='build tool for "fun"')
