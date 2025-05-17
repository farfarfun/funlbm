from funlbm.lbm import LBMD3

lbm = LBMD3(config="./config1.json")

lbm.run(max_steps=100000)
