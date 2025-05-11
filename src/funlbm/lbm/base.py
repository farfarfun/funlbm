import json
import os
import shutil
from typing import Dict, List, Union

import h5py
from funtable.kv import BaseKVTable
from funutil import deep_get, run_timer

from funlbm.base import Worker
from funlbm.config.base import BaseConfig
from funlbm.file import FileConfig, FileWrap
from funlbm.file.wrap import SaveVal
from funlbm.flow import FlowBase, FlowConfig, create_flow
from funlbm.particle import ParticleConfig, create_particle_swarm
from funlbm.util import logger, set_cpu

set_cpu()


class Config(BaseConfig):
    def __init__(
        self,
        config_path=None,
        dx=1.0,
        dt=1.0,
        max_step=10000,
        device: str = "auto",
        file=None,
        flow=None,
        particles=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dt: float = dt
        self.dx: float = dx
        self.max_step: int = max_step
        self.device: str = device
        self.file_config = FileConfig(**file)
        self.flow_config = FlowConfig(**flow)
        self.particles: List[ParticleConfig] = [
            ParticleConfig(**config) for config in particles
        ]
        self.config_path = config_path or "./config.json"

    @staticmethod
    def load_config(path: str = "./config.json") -> "Config":
        """从JSON文件加载配置

        Args:
            path: JSON配置文件路径

        Returns:
            self: 返回自身以支持链式调用
        """
        with open(path) as f:
            kwargs = {"config_path": path}
            kwargs.update(json.load(f))
            return Config(**kwargs)


def create_lbm_config(path="./config.json") -> Config:
    return Config.load_config(path)


class LBMBase(Worker):
    """格子玻尔兹曼方法的基类实现

    Args:
        flow: 流场对象
        config: 配置对象
        particle_swarm: 粒子列表
    """

    def __init__(
        self,
        config: Union[Config, str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.config: Config = (
            config if isinstance(config, Config) else Config.load_config(path=config)
        )
        self.device = self.config.device
        kwargs["device"] = self.config.device
        self.step = 0
        self.flow: FlowBase = create_flow(
            flow_config=self.config.flow_config,
            *args,
            **kwargs,
        )
        self.file_wrap = FileWrap(
            self.config.file_config,
            *args,
            **kwargs,
        )
        self.particle_swarm = create_particle_swarm(
            self.config.particles, *args, **kwargs
        )
        self.run_status = True
        self.is_save = False
        logger.info(f"Running on device: {self.device}")

    def run(self, max_steps: int = 1000000, *args, **kwargs) -> None:
        """运行模拟

        Args:
            max_steps: 最大步数
        """
        if self.step == 0:
            self.init()

        total_steps = min(max_steps, self.config.max_step)

        for i in range(total_steps):
            self.step += 1
            self.run_step(step=self.step)

            if self.run_status is False:
                break
            if self.step >= total_steps:
                break

    def _log_step_info(self, flow_track, particle_track, *args, **kwargs) -> None:
        """记录每一步的信息"""
        res = f"step={self.step:6d}"
        res += "\tf=" + ",".join([f"{i:.6f}" for i in deep_get(flow_track, "f") or []])
        res += "\tu=" + ",".join([f"{i:.6f}" for i in deep_get(flow_track, "u") or []])
        res += "\trho=" + ",".join(
            [f"{i:.6f}" for i in deep_get(flow_track, "rho") or []]
        )
        for track in particle_track:
            res += f"m={(deep_get(track, 'm') or 0):.2f}"
            res += "\tcu=" + ",".join([f"{i:.6f}" for i in deep_get(track, "cu") or []])
            res += "\tcx=" + ",".join([f"{i:.6f}" for i in deep_get(track, "cx") or []])
            res += "\tcf=" + ",".join([f"{i:.6f}" for i in deep_get(track, "cF") or []])
            res += "\tlF=" + ",".join([f"{i:.6f}" for i in deep_get(track, "lF") or []])
            res += "\tcw=" + ",".join([f"{i:.6f}" for i in deep_get(track, "cw") or []])
            res += "\tcenter=" + ",".join(
                [f"{i:.6f}" for i in deep_get(track, "coord", "center")] or []
            )
            res += "\tangle=" + ",".join(
                [f"{i:.6f}" for i in deep_get(track, "coord", "angle")] or []
            )

        logger.info(res)

    def run_step(self, *args, **kwargs) -> None:
        """执行单步模拟"""
        # 流场计算
        self._compute_flow()

        # 浸没边界处理
        self._handle_immersed_boundary()

        # 颗粒更新
        self._update_particles()

        self.save()

    def _compute_flow(self) -> None:
        """计算流场"""
        self.flow.cul_equ(step=self.step)
        self.flow.f_stream()
        self.flow.update_u_rou(step=self.step)

    @run_timer
    def _handle_immersed_boundary(self) -> None:
        """处理浸没边界"""
        self.flow_to_lagrange()
        self.particle_to_wall()

        for particle in self.particle_swarm.particles:
            particle.update_from_lar(dt=self.config.dt, gl=self.config.flow_config.gl)

        self.lagrange_to_flow()
        self.flow.cul_equ2()
        self.flow.update_u_rou()

    @run_timer
    def _update_particles(self) -> None:
        """更新粒子状态"""
        self.particle_swarm.update(dt=self.config.dt)

    def init(self, *args, **kwargs) -> None:
        self._init()

    def _init(self, *args, **kwargs):
        raise NotImplementedError()

    def flow_to_lagrange(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def lagrange_to_flow(self, n=2, h=1, *args, **kwargs):
        raise NotImplementedError()

    def particle_to_wall(self, *args, **kwargs):
        raise NotImplementedError()

    def track(self, flow_track: BaseKVTable, *args, **kwargs) -> Dict:
        _track = self.flow.track()
        flow_track.set(str(self.step), _track)
        return _track

    @run_timer
    def save(self, *args, **kwargs):
        self._log_step_info(
            self.track(self.file_wrap.track_flow),
            self.particle_swarm.track(self.step, self.file_wrap.track_particle),
        )
        self.dump_file(
            param=self.file_wrap.config.custom,
            checkpoint_path=self.file_wrap.custom_path(self.step),
            *args,
            **kwargs,
        )
        self.dump_file(
            param=self.file_wrap.config.checkpoint,
            checkpoint_path=self.file_wrap.checkpoint_path(self.step),
            *args,
            **kwargs,
        )
        if self.is_save is False:
            self.is_save = True
            if self.config.config_path != self.file_wrap.config_path:
                shutil.copy(self.config.config_path, self.file_wrap.config_path)
            self.dump_file(
                param=self.file_wrap.config.constant,
                checkpoint_path=self.file_wrap.constant_path(self.step),
                *args,
                **kwargs,
            )

    def load_checkpoint(self, checkpoint_dir="./data", *args, **kwargs):
        if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
            logger.error(f"checkpoint dir {checkpoint_dir} not exists")
            return
        file_wrap = FileWrap(config=FileConfig(cache_dir=checkpoint_dir))
        self.load_file(
            param=self.file_wrap.config.constant, file_path=file_wrap.constant_path()
        )
        self.load_file(
            param=self.file_wrap.config.checkpoint,
            file_path=file_wrap.lasted_checkpoint_path(),
        )

    def dump_file(self, param: SaveVal = None, checkpoint_path=None, *args, **kwargs):
        if param is None or checkpoint_path is None:
            return

        if self.step % param.per_step > 0:
            return

        with h5py.File(checkpoint_path, "w") as group:
            group.create_dataset("step", data=[self.step])
            self.flow.dump_file(
                group.create_group("flow"), vals=param.flow_val, *args, **kwargs
            )
            self.particle_swarm.dump_file(
                group=group.create_group("particle"),
                vals=param.particle_val,
                *args,
                **kwargs,
            )
        logger.success(
            f"save checkpoint success, step={self.step},path={checkpoint_path}"
        )

    def load_file(self, param: SaveVal = None, file_path=None, *args, **kwargs):
        if param is None or file_path is None:
            logger.error("load failed, param and checkpoint_path cannot be both None.")
            return
        if file_path is not None and os.path.exists(file_path):
            group = h5py.File(file_path, "r")
        else:
            logger.error("load failed, checkpoint_path and group cannot be both None.")
            return
        self.step = group["step"][0]
        self.flow.load_file(group.get("flow"), vals=param.flow_val, *args, **kwargs)
        self.particle_swarm.load_file(
            group=group.get("particle"), vals=param.particle_val, *args, **kwargs
        )
        logger.success(f"load checkpoint success, step={self.step},path={file_path}")
