import os

from funtecplot.dump.triangle import TriangleData
from scipy.spatial import ConvexHull

from funlbm.lbm import LBMBase
from funlbm.particle import Particle


def convert_particle(particle: Particle, filepath="particle.dat"):
    points_3d = particle.lx.cpu().numpy()
    surface_triangles = ConvexHull(points_3d).simplices
    TriangleData(
        variables=["x", "y", "z", "ux", "uy", "uz"],
        point=points_3d,
        data=particle.lu_s.cpu().numpy(),
        edge=surface_triangles + 1,
    ).dump(filepath)


def convert_particles(lbm: LBMBase, filedir="./particles"):
    os.makedirs(filedir, exist_ok=True)
    for i, particle in enumerate(lbm.particle_swarm.particles):
        convert_particle(particle, filepath=f"{filedir}/particle-{i}.dat")
