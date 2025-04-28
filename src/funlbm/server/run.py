import os


from .base import funlbm_cli

document_url = "https://darkchat.yuque.com/org-wiki-darkchat-gfaase/ul41go"


@funlbm_cli.command()
def run(config: str = "./config.json"):
    from funlbm.lbm import create_lbm

    """
    运行代码
    """
    if not os.path.exists(config):
        info = f"""配置文件不存在，访问{document_url}去配置参数吧"""
        print(info)
        raise FileExistsError(config)
    lbm = create_lbm()
    lbm.run()
