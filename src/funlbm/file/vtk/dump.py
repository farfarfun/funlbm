import numpy as np
from funlbm.lbm import LBMBase
from funtecplot.dump import PointData
from funvtk import gridToVTK, pointsToVTK


def save(lbm: LBMBase, step=10, *args, **kwargs):
    if step % lbm.config.file_config.per_steps > 0:
        return

    shape = lbm.flow.u.shape
    xf, yf, zf = np.meshgrid(
        range(shape[0]),
        range(shape[1]),
        range(shape[2]),
        indexing="ij",
        sparse=False,
    )
    flow_u = lbm.flow.u.to("cpu").numpy()
    point_data = {
        "u": (
            flow_u[:, :, :, 0],
            flow_u[:, :, :, 1],
            flow_u[:, :, :, 2],
        ),
    }

    cell_data = {
        "rho": lbm.flow.rou.to("cpu").numpy()[:, :, :, 0],
        "p": lbm.flow.p.to("cpu").numpy()[:, :, :, 0],
    }

    # t1 = self.flow.rou[10:-10, 10:-10, 10:-10, :]
    # t1 = cell_data['rho'][5:-5, 5:-5, 5:-5]
    # print("rho", step, t1.max(), t1.min(), t1.max() - t1.min())
    PointData(
        data=flow_u,
        variables=["x", "y", "z", "ux", "uy", "uz"],
        axis_dim=3,
        data_dim=3,
        title="flow_u",
    ).dump(
        filepath=f"{lbm.config.file_config.cache_dir}/tecplot_flow_u_{str(step).zfill(10)}.dat"
    )
    PointData(
        data=cell_data["rho"],
        variables=["x", "y", "z", "rho"],
        axis_dim=3,
        data_dim=1,
        title="flow_u",
    ).dump(
        filepath=f"{lbm.config.file_config.cache_dir}/tecplot_flow_rho_{str(step).zfill(10)}.dat"
    )
    PointData(
        data=cell_data["p"],
        variables=["x", "y", "z", "p"],
        axis_dim=3,
        data_dim=1,
        title="flow_u",
    ).dump(
        filepath=f"{lbm.config.file_config.cache_dir}/tecplot_flow_p_{str(step).zfill(10)}.dat"
    )

    gridToVTK(
        f"{lbm.config.file_config.cache_dir}/flow_" + str(step).zfill(10),
        xf,
        yf,
        zf,
        pointData=point_data,
        cellData=cell_data,
    )

    for i, particle in enumerate(lbm.particle_swarm.particles):
        lx = particle.lx.to("cpu").numpy()
        xf, yf, zf = lx[:, 0], lx[:, 1], lx[:, 2]
        data = {
            "u": particle.lu.to("cpu").numpy()[:, 0],
        }
        fname = f"{lbm.config.file_config.cache_dir}/particle_{str(i).zfill(3)}_{str(step).zfill(10)}"
        pointsToVTK(fname, xf, yf, zf, data=data)

    # write_to_tecplot(cell_data["rho"], f"{self.config.file_config.vtk_path}/tecplot_{str(step).zfill(10)}.dat")
