import json
import os
from enum import Enum
from typing import Any, Dict, Optional, Union

from funutil import deep_get


class BoundaryCondition(Enum):
    """边界条件类型枚举

    包含以下边界条件:
    - PERIODICAL: 周期性边界
    - WALL: 固壁边界
    - WALL_WITH_SPEED: 带速度的固壁边界
    - FAR_FIELD: 远场边界
    - NON_EQUILIBRIUM: 非平衡边界
    - NON_EQUILIBRIUM_EXREAPOLATION: 非平衡外推边界
    - FULL_DEVELOPMENT: 充分发展边界
    """

    PERIODICAL = 11000
    WALL = 1200
    WALL_WITH_SPEED = 1201
    FAR_FIELD = 1300
    NON_EQUILIBRIUM = 1400
    NON_EQUILIBRIUM_EXREAPOLATION = 1500
    FULL_DEVELOPMENT = 1600

    @classmethod
    def find(cls, code: Union[int, str]) -> "BoundaryCondition":
        """根据代码或名称查找边界条件

        Args:
            code: 边界条件代码或名称

        Returns:
            BoundaryCondition: 匹配的边界条件,默认返回WALL
        """
        try:
            if isinstance(code, int):
                return next(bc for bc in cls if bc.value == code)
            return cls[str(code)]
        except (StopIteration, KeyError):
            return cls.WALL


class BaseConfig:
    """配置基类

    提供配置的基本读写功能
    """

    def __init__(self, *args, **kwargs) -> None:
        self.expand: Dict[str, Any] = kwargs.copy()

    def _from_json(self, config_json: Dict[str, Any], **kwargs) -> None:
        """从JSON加载配置的内部方法"""
        pass

    def from_file(self, path: str) -> "BaseConfig":
        """从JSON文件加载配置

        Args:
            path: JSON配置文件路径

        Returns:
            self: 返回自身以支持链式调用
        """
        with open(path) as f:
            self.from_json(json.load(f))
        return self

    def from_json(self, config_json: Dict[str, Any], **kwargs) -> "BaseConfig":
        """从JSON字典加载配置

        Args:
            config_json: 配置字典
            **kwargs: 额外的配置参数

        Returns:
            self: 返回自身以支持链式调用
        """
        self.expand.update(kwargs)
        self.expand.update(config_json)
        self._from_json(config_json, **kwargs)
        return self

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值或默认值
        """
        return deep_get(self.expand, key) or default

    def to_json(self) -> Dict[str, Any]:
        """转换配置为JSON字典"""
        return self.expand


class Boundary(BaseConfig):
    """边界配置类

    Args:
        condition: 边界条件,默认为WALL

    属性:
        condition: 边界条件
        poiseuille: 泊肃叶流配置
    """

    def __init__(
        self, condition: BoundaryCondition = BoundaryCondition.WALL, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.condition: BoundaryCondition = condition
        self.poiseuille: Optional[Any] = None

    def is_condition(self, condition: BoundaryCondition) -> bool:
        """检查是否为指定边界条件"""
        return self.condition == condition

    def _from_json(self, config_json: Dict[str, Any], **kwargs) -> None:
        self.condition = BoundaryCondition.find(deep_get(config_json, "code") or "WALL")
        self.poiseuille = deep_get(config_json, "poiseuille")


class BoundaryConfig(BaseConfig):
    """完整边界配置类

    包含六个面的边界条件配置:
    - input: 入口边界
    - output: 出口边界
    - back: 后边界
    - forward: 前边界
    - bottom: 底边界
    - top: 顶边界
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input = Boundary(BoundaryCondition.WALL)
        self.output = Boundary(BoundaryCondition.WALL)
        self.back = Boundary(BoundaryCondition.WALL)
        self.forward = Boundary(BoundaryCondition.WALL)
        self.bottom = Boundary(BoundaryCondition.WALL)
        self.top = Boundary(BoundaryCondition.WALL)

    def _from_json(self, config_json: Dict[str, Any], **kwargs) -> None:
        for boundary in ["input", "output", "back", "forward", "bottom", "top"]:
            getattr(self, boundary).from_json(deep_get(config_json, boundary) or {})


class FileConfig(BaseConfig):
    """文件系统配置类

    属性:
        cache_dir: 缓存目录路径
        per_steps: 每隔多少步保存一次
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cache_dir: str = "./data"
        self.per_steps: int = 100

    @property
    def vtk_path(self) -> str:
        """获取VTK输出目录路径"""
        path = os.path.join(self.cache_dir, "vtk")
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def tecplot_path(self) -> str:
        """获取VTK输出目录路径"""
        path = os.path.join(self.cache_dir, "tecplot")
        os.makedirs(path, exist_ok=True)
        return path

    def _from_json(self, config_json: Dict[str, Any], **kwargs) -> None:
        self.cache_dir = deep_get(config_json, "cache_dir") or self.cache_dir
        self.per_steps = deep_get(config_json, "per_steps") or self.per_steps
