import json

from funlbm.config import Config
from funlbm.lbm.base import Solver

path = "./config.json"
_config = Config().from_json(json.loads(open(path).read()))
solver = Solver(_config)
solver.run()


"""
index-url = "http://mirrors.aliyun.com/pypi/simple/"

extra-index-url = ["https://5fc5ea58d3c4ecfbf3795547:Cy5RDDdJmr4C@packages.aliyun.com/5fc5eb115dbd287006145e5f/pypi/funpy", "http://mirrors.aliyun.com/pypi/simple/", "https://pypi.tuna.tsinghua.edu.cn/simple/", "http://pypi.mirrors.ustc.edu.cn/simple/", "https://pypi.org/simple",   "https://pypi.org/simple"]

index-strategy = "unsafe-best-match"
"""