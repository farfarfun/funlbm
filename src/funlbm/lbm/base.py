from typing import List

from funtable.kv import SQLiteStore
from funutil import deep_get

from funlbm.base import Worker
from funlbm.config.base import BaseConfig, FileConfig
from funlbm.flow import FlowConfig, FlowD3
from funlbm.particle import ParticleConfig
from funlbm.util import logger


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


class LBMBase(Worker):
    """格子玻尔兹曼方法的基类实现

    Args:
        flow: 流场对象
        config: 配置对象
        particles: 粒子列表
    """

    def __init__(
        self, flow: FlowD3, config: Config, particles: List = None, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.flow = flow
        self.config = config
        self.particles = particles or []
        self.db_store = SQLiteStore("data.db")
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

        for step in range(total_steps):
            self.run_step(step=step)
            self._log_step_info(step)
            if self.run_status is False:
                break

    def _log_step_info(self, step: int) -> None:
        """记录每一步的信息"""

        self.table_flow.set(str(step), self.flow.to_json())
        for i, particle in enumerate(self.particles):
            self.table_particle.set(str(step), str(i + 1), particle.to_json())

        info = [
            f"step={step:6d}",
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

        for particle in self.particles:
            info.append(particle.to_str(step))

        logger.info("\t".join(info))

    def run_step(self, step: int, *args, **kwargs) -> None:
        """执行单步模拟

        Args:
            step: 当前步数
        """
        # 流场计算
        self._compute_flow(step)

        # 浸没边界处理
        self._handle_immersed_boundary()

        # 颗粒更新
        self._update_particles()

        self.save(step)

    def _compute_flow(self, step: int) -> None:
        """计算流场"""
        self.flow.cul_equ(step=step)
        self.flow.f_stream()
        self.flow.update_u_rou(step=step)

    def _handle_immersed_boundary(self) -> None:
        """处理浸没边界"""
        self.flow_to_lagrange()
        self.particle_to_wall()

        for particle in self.particles:
            particle.update_from_lar(dt=self.config.dt, gl=self.config.flow_config.gl)

        self.lagrange_to_flow()
        self.flow.cul_equ2()
        self.flow.update_u_rou()

    def _update_particles(self) -> None:
        """更新粒子状态"""
        for particle in self.particles:
            particle.update(dt=self.config.dt)

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    def flow_to_lagrange(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def lagrange_to_flow(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def particle_to_wall(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, step=10, *args, **kwargs):
        raise NotImplementedError()
