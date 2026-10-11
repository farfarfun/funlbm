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

    def init(self, *args: object, **kwargs: object) -> None:
        """初始化粒子群中的全部粒子。

        Args:
            *args: 传递给粒子初始化的位置参数。
            **kwargs: 传递给粒子初始化的关键字参数。

        Returns:
            无返回值。
        """
        for particle in self.particles:
            particle.init(*args, **kwargs)

    def update(self, *args: object, **kwargs: object) -> None:
        """更新粒子群中的全部粒子状态。

        Args:
            *args: 传递给粒子更新的位置参数。
            **kwargs: 传递给粒子更新的关键字参数。

        Returns:
            无返回值。
        """
        for particle in self.particles:
            particle.update(*args, **kwargs)

    def dump_file(
        self,
        group: h5py.Group | None = None,
        vals: list[str] | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """将粒子群状态写入 HDF5 分组。

        Args:
            group: 目标 HDF5 分组。
            vals: 要写入的粒子字段；为 ``None`` 时不执行写入。
            *args: 传递给粒子持久化的位置参数。
            **kwargs: 传递给粒子持久化的关键字参数。

        Returns:
            无返回值。
        """
        if vals is None:
            return
        for i, particle in enumerate(self.particles):
            sub_group = group.create_group(f"particle_{str(i).zfill(6)}")
            particle.dump_file(sub_group, vals=vals, *args, **kwargs)

    def load_file(
        self,
        group: h5py.Group | None = None,
        vals: list[str] | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """从 HDF5 分组恢复粒子群状态。

        Args:
            group: 源 HDF5 分组。
            vals: 要读取的粒子字段；为 ``None`` 时不执行读取。
            *args: 传递给粒子反序列化的位置参数。
            **kwargs: 传递给粒子反序列化的关键字参数。

        Returns:
            无返回值。
        """
        if vals is None:
            return
        for i, particle in enumerate(self.particles):
            sub_group = group.get(f"particle_{str(i).zfill(6)}")
            particle.load_file(sub_group, vals=vals, *args, **kwargs)

    def track(
        self,
        step: int,
        particle_track: BaseKKVTable,
        *args: object,
        **kwargs: object,
    ) -> list[dict]:
        """记录每个粒子在当前步的跟踪数据。

        Args:
            step: 当前模拟步数。
            particle_track: 粒子跟踪数据表。
            *args: 为扩展逻辑保留的位置参数。
            **kwargs: 为扩展逻辑保留的关键字参数。

        Returns:
            各粒子跟踪数据组成的列表。
        """
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
