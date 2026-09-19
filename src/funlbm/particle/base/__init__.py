from .base import Particle, ParticleConfig
from .ellipsoid import Ellipsoid
from .sphere import Sphere
from .spheroid import Spheroid

__all__ = ["Ellipsoid", "Particle", "Sphere", "Spheroid", "create_particle"]


def create_particle(config: ParticleConfig, device="cpu") -> Particle:
    if config.type == "ellipsoid":
        return Ellipsoid(config=config, device=device)
    elif config.type == "spheroid":
        return Spheroid(config=config, device=device)
    else:
        return Sphere(config=config, device=device)
