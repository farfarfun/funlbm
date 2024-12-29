from typing import List

import numpy as np
from funutil import deep_get

from funlbm.base import Worker
from funlbm.config.base import BaseConfig, FileConfig
from funlbm.flow import FlowConfig, FlowD3
from funlbm.particle import ParticleConfig
from funlbm.util import logger


class Config(BaseConfig):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dt = 1
        self.dx = 1
        self.device = "cpu"
        self.file_config = FileConfig()
        self.flow_config = FlowConfig()
        self.particles: List[ParticleConfig] = []

    def _from_json(self, config_json: dict, *args, **kwargs):
        self.dt = deep_get(config_json, "dt") or self.dt
        self.dx = deep_get(config_json, "dx") or self.dx
        self.device = deep_get(config_json, "device") or self.device
        self.file_config = FileConfig().from_json(deep_get(config_json, "file") or {})
        self.flow_config = FlowConfig().from_json(deep_get(config_json, "flow") or {})
        for config in deep_get(config_json, "particles") or []:
            self.particles.append(ParticleConfig().from_json(config_json=config))


class LBMBase(Worker):
    def __init__(self, flow: FlowD3, config: Config, particles=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow = flow
        self.config = config
        self.particles = particles

        logger.info(f"device:={self.device}")

    def run(self, max_steps=1000000, *args, **kwargs):
        self.init()
        total = int(max_steps / self.config.dt)
        # pbar = tqdm(range(total))
        pbar = range(total)
        for step in pbar:
            self.run_step(step=step)

            res = f"step={step}"
            res += f"\tf_max={np.max(self.flow.f.numpy()):8f}"
            res += f"\tu_max={np.max(self.flow.u.numpy()):8f}"
            for particle in self.particles:
                res += "\t"
                res += particle.to_str(step)
            logger.info(res)

    def init(self, *args, **kwargs):
        raise NotImplementedError()

    def run_step(self, step, *args, **kwargs):
        # 流场碰撞
        self.flow.cul_equ(step=step)
        # 流场迁移
        self.flow.f_stream()
        # 流场计算-速度&密度
        self.flow.update_u_rou(step=step)

        # 浸没计算-流场->拉格朗日点
        self.flow_to_lagrange()
        self.particle_to_wall()

        # 浸没计算-拉格朗日点->颗粒
        [
            particle.update_from_lar(dt=self.config.dt, gl=self.config.flow_config.gl)
            for particle in self.particles
        ]

        # 浸没计算-拉格朗日点->流场
        self.lagrange_to_flow()

        # 流场二次碰撞
        self.flow.cul_equ2()
        # 流场计算-速度&密度
        self.flow.update_u_rou()

        # 颗粒->拉格朗日点
        [particle.update(dt=self.config.dt) for particle in self.particles]

        self.save(step)

    def flow_to_lagrange(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def lagrange_to_flow(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def particle_to_wall(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, step=10, *args, **kwargs):
        raise NotImplementedError()
