from funlbm.lbm import LBMD3, Config

lbm = LBMD3(config=Config().from_file("./config.json"))
lbm.run()

# pip install funlbm -U -i  https://pypi.org/simple
