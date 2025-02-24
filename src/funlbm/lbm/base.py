from typing import List

import h5py
from funtable.kv import SQLiteStore
from funutil import deep_get

from funlbm.base import Worker
from funlbm.config.base import BaseConfig, FileConfig
from funlbm.flow import FlowConfig, FlowD3, create_flow
from funlbm.particle import ParticleConfig, ParticleSwarm, create_particle_swarm
from funlbm.util import logger, set_cpu

set_cpu()


class Config(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt: float = 1.0
        self.dx: float = 1.0
        self.max_step = 10000
        self.device: str = "auto"
        self.file_config = FileConfig()
        self.flow_config = FlowConfig()

        self.particles: List[ParticleConfig] = []

    def _from_json(self, config_json: dict, *args, **kwargs) -> "Config":
        self.dt = deep_get(config_json, "dt") or self.dt
        self.dx = deep_get(config_json, "dx") or self.dx
        self.max_step = deep_get(config_json, "max_step") or self.max_step
        self.device = deep_get(config_json, "device") or self.device
        self.file_config = FileConfig().from_json(deep_get(config_json, "file") or {})
        self.flow_config = FlowConfig().from_json(deep_get(config_json, "flow") or {})
        for config in deep_get(config_json, "particles") or []:
            self.particles.append(ParticleConfig().from_json(config_json=config))
        return self


def create_lbm_config(path="./config.json") -> Config:
    return Config().from_file(path)


class LBMBase(Worker):
    """格子玻尔兹曼方法的基类实现

    Args:
        flow: 流场对象
        config: 配置对象
        particle_swarm: 粒子列表
    """

    def __init__(
        self,
        config: Config = None,
        flow: FlowD3 = None,
        particle_swarm: ParticleSwarm = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config: Config = config or create_lbm_config()

        self.step = 1
        self.flow = flow or create_flow(
            flow_config=self.config.flow_config,
            device=self.device,
            *args,
            **kwargs,
        )
        self.particle_swarm = particle_swarm or create_particle_swarm(
            self.config.particles, device=self.device
        )

        self.db_store = SQLiteStore("funlbm-global.db")
        self.db_store.create_kv_table("flow")
        self.db_store.create_kkv_table("particle")
        self.table_flow = self.db_store.get_table("flow")
        self.table_particle = self.db_store.get_table("particle")
        self.run_status = True
        logger.info(f"Running on device: {self.device}")

    def run(self, max_steps: int = 1000000, *args, **kwargs) -> None:
        """运行模拟

        Args:
            max_steps: 最大步数
        """
        self.init()
        total_steps = min(max_steps, self.config.max_step)

        for self.step in range(total_steps):
            self.run_step(step=self.step)
            self._log_step_info(self.step)
            if self.run_status is False:
                break

    def _log_step_info(self, *args, **kwargs) -> None:
        """记录每一步的信息"""

        self.table_flow.set(str(self.step), self.flow.to_json())
        for i, particle in enumerate(self.particle_swarm.particles):
            self.table_particle.set(str(self.step), str(i + 1), particle.to_json())

        info = [
            f"step={self.step:6d}",
            "f="
            + ",".join(
                [
                    f"{i:.6f}"
                    for i in [self.flow.f.min(), self.flow.f.mean(), self.flow.f.max()]
                ]
            ),
            "u="
            + ",".join(
                [
                    f"{i:.6f}"
                    for i in [self.flow.u.min(), self.flow.u.mean(), self.flow.u.max()]
                ]
            ),
            "rho="
            + ",".join(
                [
                    f"{i:.6f}"
                    for i in [
                        self.flow.rou.min(),
                        self.flow.rou.mean(),
                        self.flow.rou.max(),
                    ]
                ]
            ),
        ]

        for particle in self.particle_swarm.particles:
            info.append(particle.to_str(self.step))

        logger.info("\t".join(info))

    def run_step(self, *args, **kwargs) -> None:
        """执行单步模拟"""
        # 流场计算
        self._compute_flow()

        # 浸没边界处理
        self._handle_immersed_boundary()

        # 颗粒更新
        self._update_particles()

        self.save(self.step)

    def _compute_flow(self) -> None:
        """计算流场"""
        self.flow.cul_equ(step=self.step)
        self.flow.f_stream()
        self.flow.update_u_rou(step=self.step)

    def _handle_immersed_boundary(self) -> None:
        """处理浸没边界"""
        self.flow_to_lagrange()
        self.particle_to_wall()

        for particle in self.particle_swarm.particles:
            particle.update_from_lar(dt=self.config.dt, gl=self.config.flow_config.gl)

        self.lagrange_to_flow()
        self.flow.cul_equ2()
        self.flow.update_u_rou()

    def _update_particles(self) -> None:
        """更新粒子状态"""
        self.particle_swarm.update(dt=self.config.dt)

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    def flow_to_lagrange(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def lagrange_to_flow(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def particle_to_wall(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, step=10, *args, **kwargs):
        if step % self.config.file_config.per_steps > 0:
            return
        filepath = f"{self.config.file_config.checkpoint_path}/checkpoint_{str(step).zfill(10)}.h5"
        with h5py.File(filepath, "w") as fw:
            self.dump_checkpoint(fw)

    def dump_checkpoint(self, group: h5py.Group = None, *args, **kwargs):
        self.flow.dump_checkpoint(group.create_group("flow"), *args, **kwargs)
        self.particle_swarm.dump_checkpoint(
            group=group.create_group("particle"), *args, **kwargs
        )
