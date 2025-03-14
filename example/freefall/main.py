import os.path

from funlbm.file.tecplot import convert_particle
from funlbm.lbm import create_lbm, create_lbm_by_checkpoint

# pip install funlbm -U -i  https://pypi.org/simple
if os.path.exists("./data2") and True:
    lbm = create_lbm_by_checkpoint("./data")
else:
    lbm = create_lbm()


lbm.run(max_steps=100000)

convert_particle(lbm.particle_swarm.particles[0])
