import json
import os
import shutil
from typing import Any

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
        config_path: str | None = None,
        dx: float = 1.0,
        dt: float = 1.0,
        max_step: int = 10000,
        device: str = "auto",
        file: dict[str, Any] | None = None,
        flow: dict[str, Any] | None = None,
        particles: list[dict[str, Any]] | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """构造 LBM 运行配置。

        Args:
            config_path: 配置文件路径；未指定时使用 ``./config.json``。
            dx: 格点空间步长。
            dt: 时间步长。
            max_step: 模拟允许执行的最大步数。
            device: PyTorch 计算设备名称。
            file: 文件输出配置。
            flow: 流场配置。
            particles: 粒子配置列表。
            *args: 传递给配置基类的位置参数。
            **kwargs: 未识别的扩展配置。
        """
        super().__init__(*args, **kwargs)
        self.dt: float = dt
        self.dx: float = dx
        self.max_step: int = max_step
        self.device: str = device
        self.file_config = FileConfig(**(file or {}))
        self.flow_config = FlowConfig(**(flow or {}))
        self.particles: list[ParticleConfig] = [
            ParticleConfig(**config) for config in particles or []
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


def create_lbm_config(path: str = "./config.json") -> Config:
    """从 JSON 配置文件创建 `Config` 对象。

    Args:
        path: JSON 配置文件路径，默认为当前目录下的 config.json。

    Returns:
        根据配置文件内容构造出的 `Config` 实例。

    Raises:
        FileNotFoundError: `path` 指向的文件不存在时抛出。
    """
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
        config: Config | str | None = None,
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
        """初始化流场、粒子和求解器状态。

        Args:
            *args: 传递给具体求解器初始化实现的位置参数。
            **kwargs: 传递给具体求解器初始化实现的关键字参数。

        Returns:
            无返回值。
        """
        self._init()

    def _init(self, *args, **kwargs):
        raise NotImplementedError()

    def flow_to_lagrange(
        self, n: int = 2, h: float = 1, *args: object, **kwargs: object
    ) -> None:
        """将欧拉流场量插值到拉格朗日粒子点。

        Args:
            n: 插值核的半宽。
            h: 插值核的格点间距。
            *args: 子类扩展的位置参数。
            **kwargs: 子类扩展的关键字参数。

        Returns:
            无返回值。

        Raises:
            NotImplementedError: 子类未实现耦合计算时抛出。
        """
        raise NotImplementedError()

    def lagrange_to_flow(
        self, n: int = 2, h: float = 1, *args: object, **kwargs: object
    ) -> None:
        """将粒子作用力扩散回欧拉流场。

        Args:
            n: 扩散核的半宽。
            h: 扩散核的格点间距。
            *args: 子类扩展的位置参数。
            **kwargs: 子类扩展的关键字参数。

        Returns:
            无返回值。

        Raises:
            NotImplementedError: 子类未实现耦合计算时抛出。
        """
        raise NotImplementedError()

    def particle_to_wall(self, *args: object, **kwargs: object) -> None:
        """计算粒子边界对流场的作用。

        Args:
            *args: 子类扩展的位置参数。
            **kwargs: 子类扩展的关键字参数。

        Returns:
            无返回值。

        Raises:
            NotImplementedError: 子类未实现边界耦合时抛出。
        """
        raise NotImplementedError()

    def track(self, flow_track: BaseKVTable, *args, **kwargs) -> dict:
        _track = self.flow.track()
        flow_track.set(str(self.step), _track)
        return _track

    @run_timer
    def save(self, *args: object, **kwargs: object) -> None:
        """按输出配置保存跟踪数据和检查点。

        Args:
            *args: 传递给序列化实现的位置参数。
            **kwargs: 传递给序列化实现的关键字参数。

        Returns:
            无返回值。
        """
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

    def load_checkpoint(
        self, checkpoint_dir: str = "./data", *args: object, **kwargs: object
    ) -> None:
        """从目录中的最新检查点恢复流场和粒子状态。

        Args:
            checkpoint_dir: 包含 ``constant.h5`` 和检查点目录的输出目录。
            *args: 传递给加载实现的位置参数。
            **kwargs: 传递给加载实现的关键字参数。

        Returns:
            无返回值；目录不存在时仅记录错误日志。
        """
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

    def dump_file(
        self,
        param: SaveVal | None = None,
        checkpoint_path: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """将当前求解器状态写入 HDF5 文件。

        Args:
            param: 要写入字段及频率的配置。
            checkpoint_path: 目标 HDF5 文件路径。
            *args: 传递给流场和粒子序列化的位置参数。
            **kwargs: 传递给流场和粒子序列化的关键字参数。

        Returns:
            无返回值；保存频率未到或参数不完整时直接跳过。
        """
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

    def load_file(
        self,
        param: SaveVal | None = None,
        file_path: str | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        """从 HDF5 文件加载流场和粒子状态。

        Args:
            param: 要读取字段的配置。
            file_path: 来源 HDF5 文件路径。
            *args: 传递给流场和粒子反序列化的位置参数。
            **kwargs: 传递给流场和粒子反序列化的关键字参数。

        Returns:
            无返回值；参数不完整或文件不存在时仅记录错误日志。
        """
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
