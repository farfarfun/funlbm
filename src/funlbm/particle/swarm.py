from typing import List

import h5py

from funlbm.base import Worker
from funlbm.particle import Particle, ParticleConfig
from funlbm.util import logger

from .base import create_particle


class ParticleSwarm(Worker):
    def __init__(self, configs: List[ParticleConfig] = [], device="cpu"):
        self.particles: List[Particle] = []
        for config in configs:
            self.particles.append(create_particle(config, device=device))

    def init(self, *args, **kwargs):
        for particle in self.particles:
            particle.init(*args, **kwargs)
        self.save_particle()

    def update(self, *args, **kwargs):
        for particle in self.particles:
            particle.update(*args, **kwargs)

    def save_particle(self):
        with h5py.File("./funlbm-particle-swarm-lagrange.h5", "w") as fw:
            for i, particle in enumerate(self.particles):
                fw.create_dataset(f"particle{i}", data=particle._lagrange.cpu().numpy())
        logger.success("save particle lagrange success.")

    def dump_checkpoint(self, group: h5py.Group = None, *args, **kwargs):
        for i, particle in enumerate(self.particles):
            sub_group = group.create_group(f"particle_{str(i).zfill(6)}")
            particle.dump_checkpoint(sub_group, *args, **kwargs)

    def load_checkpoint(self, group: h5py.Group = None, *args, **kwargs):
        for i, particle in enumerate(self.particles):
            sub_group = group.get(f"particle_{str(i).zfill(6)}")
            particle.load_checkpoint(sub_group, *args, **kwargs)


def create_particle_swarm(
    configs: List[ParticleConfig] = [], device="cpu", *args, **kwargs
):
    return ParticleSwarm(configs=configs, device=device, *args, **kwargs)
