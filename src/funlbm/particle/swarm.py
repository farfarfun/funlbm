from typing import List

import h5py
from funlbm.particle import Particle, ParticleConfig
from .base import create_particle


class ParticleSwarm:
    def __init__(self, configs=[]):
        self.particles: List[Particle] = []
        for config in configs:
            self.particles.append(
                create_particle(ParticleConfig().from_json(config_json=config))
            )

    def init(self, *args, **kwargs):
        for particle in self.particles:
            particle.init(*args, **kwargs)

    def update(self, *args, **kwargs):
        for particle in self.particles:
            particle.update(*args, **kwargs)

    def save_particle(self):
        with h5py.File("./particle_lagrange.h5", "w") as fw:
            for i, particle in enumerate(self.particles):
                fw.create_dataset(f"particle{i}", particle._lagrange)
