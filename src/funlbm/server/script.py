import argparse
import os

from funlbm.lbm import LBMD3, Config

document_url = "https://darkchat.yuque.com/org-wiki-darkchat-gfaase/ul41go"


def work(config_file="./config.json"):
    if not os.path.exists(config_file):
        info = f"""配置文件不存在，访问{document_url}去配置参数吧"""
        print(info)
        raise FileExistsError(config_file)
    config = Config().from_file(config_file)
    lbm = LBMD3(config=config, device=config.device)
    lbm.run()


def funlbm():
    parser = argparse.ArgumentParser(prog="PROG")

    # 添加子命令
    parser.add_argument("url", help="下载链接")
    parser.add_argument(
        "--config",
        default="./config.json",
        type=str,
        help=f"配置文件，参考{document_url}",
    )

    parser.set_defaults(func=work)

    args, unknown = parser.parse_known_args()
    args.func(args)
