from typing import List, Optional, Union

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from funlbm.base import Worker
from funlbm.config.base import BaseConfig
from funlbm.util import tensor_format


class CoordConfig(BaseConfig):
    """坐标系统配置类

    Args:
        alpha: 绕x轴旋转角度,默认π/2
        beta: 绕y轴旋转角度,默认0
        gamma: 绕z轴旋转角度,默认0
    """

    def __init__(
        self,
        center=None,
        alpha: float = np.pi / 2,
        beta: float = 0,
        gamma: float = 0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.center: List[float] = center or [0, 0, 0]  # 坐标系中心点
        self.alpha, self.beta, self.gamma = alpha, beta, gamma  # 三个旋转角度


class Coordinate(Worker):
    """3D坐标系统类

    支持坐标系的旋转和平移操作

    属性:
        center (Tensor): 坐标系原点位置
        w (Tensor): 旋转角度[alpha, beta, gamma]
        rotation (R): 旋转矩阵对象

    Args:
        config (CoordConfig): 坐标系配置对象
    """

    def __init__(self, config: Optional[CoordConfig] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = config or CoordConfig()
        self.center = torch.tensor(
            config.center, device=self.device, dtype=torch.float32
        )
        self.angle = torch.tensor(
            [config.alpha, config.beta, config.gamma],
            device=self.device,
            dtype=torch.float32,
        )
        self.rotation = R.from_rotvec(self.angle.cpu().numpy())

    def cul_point(
        self, points: Union[List[float], np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """计算点在旋转和平移后的新位置

        Args:
            points: 输入点坐标,可以是单个点[x,y,z]或点集[N,3]

        Returns:
            Tensor: 变换后的点坐标
        """
        if isinstance(points, (list, tuple)):
            points = np.array(points)
        elif isinstance(points, torch.Tensor):
            points = points.cpu().numpy()

        return (
            torch.tensor(
                self.rotation.apply(points), device=self.device, dtype=torch.float32
            )
            + self.center
        )

    def update(self, center: torch.Tensor, w: torch.Tensor) -> None:
        """更新旋转角度并重新计算旋转矩阵

        Args:
            cw: 旋转角度的变化量[d_alpha, d_beta, d_gamma]

        Raises:
            ValueError: 当cw为None或形状不正确时
            TypeError: 当cw类型不支持时
        """
        self.center = center
        self.angle += w
        self.rotation = R.from_rotvec(self.angle.cpu().numpy())

    def to_json(self):
        return {
            "center": tensor_format(self.center),
            "angle": tensor_format(self.angle),
        }

    def dump_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if group is None:
            return
        vals = vals or ["center", "angle"]
        if "center" in vals:
            self.dump_dataset(group, "center", self.center)
        if "angle" in vals:
            self.dump_dataset(group, "angle", self.angle)

    def load_file(self, group: h5py.Group = None, vals=None, *args, **kwargs):
        if group is None:
            return
        vals = vals or group.keys()
        if "center" in vals:
            self.center = self.load_dataset(group, "center")
        if "angle" in vals:
            self.angle = self.load_dataset(group, "angle")


def example():
    """Example usage of the Coordinate class."""
    coord = Coordinate(CoordConfig(alpha=np.pi / 2.0, beta=0, gamma=0))
    result = coord.cul_point([1, 0, 1])
    print(f"Transformed point: {result}")


# example()
