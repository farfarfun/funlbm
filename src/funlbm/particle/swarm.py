from typing import List

import h5py

from funlbm.particle import Particle, ParticleConfig
from funlbm.util import logger

from .base import create_particle


class ParticleSwarm:
    def __init__(self, configs=[], device="cpu"):
        self.particles: List[Particle] = []
        for config in configs:
            self.particles.append(
                create_particle(ParticleConfig(config_json=config), device=device)
            )

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
