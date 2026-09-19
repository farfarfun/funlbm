import h5py
from funtable.kv import BaseKKVTable

from funlbm.base import Worker
from funlbm.particle import Particle, ParticleConfig

from .base import create_particle


class ParticleSwarm(Worker):
    """粒子群，管理一组粒子实例的初始化、更新与持久化"""

    def __init__(self, configs: list[ParticleConfig] | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.particles: list[Particle] = []
        for config in configs or []:
            self.particles.append(create_particle(config, device=self.device))

    def init(self, *args, **kwargs):
        for particle in self.particles:
            particle.init(*args, **kwargs)

    def update(self, *args, **kwargs):
        for particle in self.particles:
            particle.update(*args, **kwargs)

    def dump_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if vals is None:
            return
        for i, particle in enumerate(self.particles):
            sub_group = group.create_group(f"particle_{str(i).zfill(6)}")
            particle.dump_file(sub_group, vals=vals, *args, **kwargs)

    def load_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if vals is None:
            return
        for i, particle in enumerate(self.particles):
            sub_group = group.get(f"particle_{str(i).zfill(6)}")
            particle.load_file(sub_group, vals=vals, *args, **kwargs)

    def track(self, step, particle_track: BaseKKVTable, *args, **kwargs) -> list[dict]:
        res = []
        for i, particle in enumerate(self.particles):
            _track = particle.track()
            particle_track.set(str(step), str(i + 1), _track)
            res.append(_track)
        return res


def create_particle_swarm(
    configs: list[ParticleConfig] | None = None, device="cpu", *args, **kwargs
) -> ParticleSwarm:
    """创建粒子群实例

    Args:
        configs: 粒子配置列表，默认为空列表
        device: 计算设备，默认为 cpu

    Returns:
        ParticleSwarm: 初始化好的粒子群
    """
    return ParticleSwarm(configs=configs or [], device=device, *args, **kwargs)
