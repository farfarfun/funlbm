from .base import Particle, ParticleConfig
from .ellipsoid import Ellipsoid
from .sphere import Sphere

__all__ = ["Particle", "Particle", "create_particle", "Ellipsoid", "Sphere"]


def create_particle(config: ParticleConfig, device="cpu") -> Particle:
    if config.type == "ellipsoid":
        return Ellipsoid(config=config, device=device)
    else:
        return Sphere(config=config, device=device)
