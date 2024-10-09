import json

from funlbm2.config import Config
from funlbm2.lbm.base import Solver

path = './config.json'
_config = Config().from_json(json.loads(open(path).read()))
solver = Solver(_config)
solver.run()
