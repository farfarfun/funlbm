import matplotlib

matplotlib.use("TkAgg")

from funlbm.lbm import LBMBase
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from funlbm.particle import Particle


def plot_particle(particle: Particle):
    points_3d = particle.lx.cpu().numpy()
    surface_triangles = ConvexHull(points_3d).simplices
    # 绘制结果
    fig = plt.figure(dpi=100, figsize=[10, 10])
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_trisurf(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        triangles=surface_triangles,
        cmap="viridis",
    )
    # 设置X轴和Y轴比例一致
    ax.set_box_aspect([1, 1, 1])
    ax.scatter(
        [points_3d.min(), points_3d.max()],
        [points_3d.min(), points_3d.max()],
        [points_3d.min(), points_3d.max()],
        s=1,
    )

    # ax = fig.add_subplot(212, projection="3d")
    # ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
    plt.show()


def plot_particles(lbm: LBMBase):
    for particle in lbm.particle_swarm.particles:
        plot_particle(particle)
