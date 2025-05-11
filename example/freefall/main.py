from funlbm.lbm import LBMD3

lbm = LBMD3(config="./config4.json")

lbm.run(max_steps=100000)
