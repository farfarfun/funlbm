from .base import Particle, ParticleConfig
from .ellipsoid import Ellipsoid
from .sphere import Sphere
from .spheroid import Spheroid
from torch import device as TorchDevice

__all__ = ["Ellipsoid", "Particle", "Sphere", "Spheroid", "create_particle"]


def create_particle(
    config: ParticleConfig, device: str | TorchDevice = "cpu"
) -> Particle:
    """根据配置创建对应形状的粒子。

    Args:
        config: 粒子的形状和物理参数配置。
        device: 执行张量计算的设备名称或 PyTorch 设备对象。

    Returns:
        与 ``config.type`` 对应的粒子实例。
    """
    if config.type == "ellipsoid":
        return Ellipsoid(config=config, device=device)
    elif config.type == "spheroid":
        return Spheroid(config=config, device=device)
    else:
        return Sphere(config=config, device=device)
