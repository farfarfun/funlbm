import os

from funtecplot.dump.triangle import TriangleData
from scipy.spatial import ConvexHull

from funlbm.lbm import LBMBase
from funlbm.particle import Particle


def convert_particle(particle: Particle, filepath: str = "particle.dat") -> None:
    """将单个粒子导出为 Tecplot 三角网格数据文件。

    Args:
        particle: 待导出的粒子对象。
        filepath: 输出 Tecplot 文件路径。

    Returns:
        无返回值。
    """
    points_3d = particle.lx.cpu().numpy()
    surface_triangles = ConvexHull(points_3d).simplices
    TriangleData(
        variables=["x", "y", "z", "ux", "uy", "uz"],
        point=points_3d,
        data=particle.lu_s.cpu().numpy(),
        edge=surface_triangles + 1,
    ).dump(filepath)


def convert_particles(lbm: LBMBase, filedir: str = "./particles") -> None:
    """将求解器中的全部粒子导出到指定目录。

    Args:
        lbm: 包含粒子群的 LBM 求解器。
        filedir: 输出目录。

    Returns:
        无返回值。
    """
    os.makedirs(filedir, exist_ok=True)
    for i, particle in enumerate(lbm.particle_swarm.particles):
        convert_particle(particle, filepath=f"{filedir}/particle-{i}.dat")
