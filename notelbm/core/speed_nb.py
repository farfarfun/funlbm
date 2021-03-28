from numba import jit


@jit(nopython=True, cache=True)
def nb_zou_he_left_wall_velocity(lx, ly, u, u_left, rho, g):
    # Zou-He left wall velocity b.c.
    cst1 = 2.0 / 3.0
    cst2 = 1.0 / 6.0
    cst3 = 1.0 / 2.0

    u[0, 0, :] = u_left[0, :]
    u[1, 0, :] = u_left[1, :]

    rho[0, :] = (g[0, 0, :] + g[3, 0, :] + g[4, 0, :] +
                 2.0 * g[2, 0, :] + 2.0 * g[6, 0, :] +
                 2.0 * g[7, 0, :]) / (1.0 - u[0, 0, :])

    g[1, 0, :] = (g[2, 0, :] + cst1 * rho[0, :] * u[0, 0, :])

    g[5, 0, :] = (g[6, 0, :] - cst3 * (g[3, 0, :] - g[4, 0, :]) +
                  cst2 * rho[0, :] * u[0, 0, :] +
                  cst3 * rho[0, :] * u[1, 0, :])

    g[8, 0, :] = (g[7, 0, :] + cst3 * (g[3, 0, :] - g[4, 0, :]) +
                  cst2 * rho[0, :] * u[0, 0, :] -
                  cst3 * rho[0, :] * u[1, 0, :])


@jit(nopython=True, cache=True)
def nb_zou_he_right_wall_velocity(lx, ly, u, u_right, rho, g):
    # Zou-He right wall velocity b.c.
    cst1 = 2.0 / 3.0
    cst2 = 1.0 / 6.0
    cst3 = 1.0 / 2.0

    u[0, lx, :] = u_right[0, :]
    u[1, lx, :] = u_right[1, :]

    rho[lx, :] = (g[0, lx, :] + g[3, lx, :] + g[4, lx, :] +
                  2.0 * g[1, lx, :] + 2.0 * g[5, lx, :] +
                  2.0 * g[8, lx, :]) / (1.0 + u[0, lx, :])

    g[2, lx, :] = (g[1, lx, :] - cst1 * rho[lx, :] * u[0, lx, :])

    g[6, lx, :] = (g[5, lx, :] + cst3 * (g[3, lx, :] - g[4, lx, :]) -
                   cst2 * rho[lx, :] * u[0, lx, :] -
                   cst3 * rho[lx, :] * u[1, lx, :])

    g[7, lx, :] = (g[8, lx, :] - cst3 * (g[3, lx, :] - g[4, lx, :]) -
                   cst2 * rho[lx, :] * u[0, lx, :] +
                   cst3 * rho[lx, :] * u[1, lx, :])


@jit(nopython=True, cache=True)
def nb_zou_he_right_wall_pressure(lx, ly, u, rho_right, u_right, rho, g):
    # Zou-He right wall pressure b.c.
    cst1 = 2.0 / 3.0
    cst2 = 1.0 / 6.0
    cst3 = 1.0 / 2.0

    rho[lx, :] = rho_right[:]
    u[1, lx, :] = u_right[1, :]

    u[0, lx, :] = (g[0, lx, :] + g[3, lx, :] + g[4, lx, :] +
                   2.0 * g[1, lx, :] + 2.0 * g[5, lx, :] +
                   2.0 * g[8, lx, :]) / rho[lx, :] - 1.0

    g[2, lx, :] = (g[1, lx, :] - cst1 * rho[lx, :] * u[0, lx, :])

    g[6, lx, :] = (g[5, lx, :] + cst3 * (g[3, lx, :] - g[4, lx, :]) -
                   cst2 * rho[lx, :] * u[0, lx, :] -
                   cst3 * rho[lx, :] * u[1, lx, :])

    g[7, lx, :] = (g[8, lx, :] - cst3 * (g[3, lx, :] - g[4, lx, :]) -
                   cst2 * rho[lx, :] * u[0, lx, :] +
                   cst3 * rho[lx, :] * u[1, lx, :])


@jit(nopython=True, cache=True)
def nb_zou_he_top_wall_velocity(lx, ly, u, u_top, rho, g):
    # Zou-He no-slip top wall velocity b.c.
    cst1 = 2.0 / 3.0
    cst2 = 1.0 / 6.0
    cst3 = 1.0 / 2.0

    u[0, :, ly] = u_top[0, :]
    u[1, :, ly] = u_top[1, :]

    rho[:, 0] = (g[0, :, 0] + g[1, :, 0] + g[2, :, 0] +
                 2.0 * g[3, :, 0] + 2.0 * g[5, :, 0] +
                 2.0 * g[7, :, 0]) / (1.0 + u[1, :, ly])

    g[4, :, ly] = (g[3, :, ly] - cst1 * rho[:, ly] * u[1, :, ly])

    g[8, :, ly] = (g[7, :, ly] - cst3 * (g[1, :, ly] - g[2, :, ly]) +
                   cst3 * rho[:, ly] * u[0, :, ly] -
                   cst2 * rho[:, ly] * u[1, :, ly])

    g[6, :, ly] = (g[5, :, ly] + cst3 * (g[1, :, ly] - g[2, :, ly]) -
                   cst3 * rho[:, ly] * u[0, :, ly] -
                   cst2 * rho[:, ly] * u[1, :, ly])


@jit(nopython=True, cache=True)
def nb_zou_he_bottom_wall_velocity(lx, ly, u, u_bot, rho, g):
    # Zou-He no-slip bottom wall velocity b.c.
    cst1 = 2.0 / 3.0
    cst2 = 1.0 / 6.0
    cst3 = 1.0 / 2.0

    u[0, :, 0] = u_bot[0, :]
    u[1, :, 0] = u_bot[1, :]

    rho[:, 0] = (g[0, :, 0] + g[1, :, 0] + g[2, :, 0] +
                 2.0 * g[4, :, 0] + 2.0 * g[6, :, 0] +
                 2.0 * g[8, :, 0]) / (1.0 - u[1, :, 0])

    g[3, :, 0] = (g[4, :, 0] + cst1 * rho[:, 0] * u[1, :, 0])

    g[5, :, 0] = (g[6, :, 0] - cst3 * (g[1, :, 0] - g[2, :, 0]) +
                  cst3 * rho[:, 0] * u[0, :, 0] +
                  cst2 * rho[:, 0] * u[1, :, 0])

    g[7, :, 0] = (g[8, :, 0] + cst3 * (g[1, :, 0] - g[2, :, 0]) -
                  cst3 * rho[:, 0] * u[0, :, 0] +
                  cst2 * rho[:, 0] * u[1, :, 0])


@jit(nopython=True, cache=True)
def nb_zou_he_bottom_left_corner_velocity(lx, ly, u, rho, g):
    # Zou-He no-slip bottom left corner velocity b.c.
    u[0, 0, 0] = u[0, 1, 0]
    u[1, 0, 0] = u[1, 1, 0]

    rho[0, 0] = rho[1, 0]

    g[1, 0, 0] = (g[2, 0, 0] + (2.0 / 3.0) * rho[0, 0] * u[0, 0, 0])

    g[3, 0, 0] = (g[4, 0, 0] + (2.0 / 3.0) * rho[0, 0] * u[1, 0, 0])

    g[5, 0, 0] = (g[6, 0, 0] + (1.0 / 6.0) * rho[0, 0] * u[0, 0, 0]
                  + (1.0 / 6.0) * rho[0, 0] * u[1, 0, 0])

    g[7, 0, 0] = 0.0
    g[8, 0, 0] = 0.0

    g[0, 0, 0] = (rho[0, 0]
                  - g[1, 0, 0] - g[2, 0, 0] - g[3, 0, 0] - g[4, 0, 0]
                  - g[5, 0, 0] - g[6, 0, 0] - g[7, 0, 0] - g[8, 0, 0])


@jit(nopython=True, cache=True)
def nb_zou_he_top_left_corner_velocity(lx, ly, u, rho, g):
    # Zou-He no-slip top left corner velocity b.c.
    u[0, 0, ly] = u[0, 1, ly]
    u[1, 0, ly] = u[1, 1, ly]

    rho[0, ly] = rho[1, ly]

    g[1, 0, ly] = (g[2, 0, ly] + (2.0 / 3.0) * rho[0, ly] * u[0, 0, ly])

    g[4, 0, ly] = (g[3, 0, ly] - (2.0 / 3.0) * rho[0, ly] * u[1, 0, ly])

    g[8, 0, ly] = (g[7, 0, ly] + (1.0 / 6.0) * rho[0, ly] * u[0, 0, ly]
                   - (1.0 / 6.0) * rho[0, ly] * u[1, 0, ly])

    g[5, 0, ly] = 0.0
    g[6, 0, ly] = 0.0

    g[0, 0, ly] = (rho[0, ly]
                   - g[1, 0, ly] - g[2, 0, ly] - g[3, 0, ly] - g[4, 0, ly]
                   - g[5, 0, ly] - g[6, 0, ly] - g[7, 0, ly] - g[8, 0, ly])


@jit(nopython=True, cache=True)
def nb_zou_he_top_right_corner_velocity(lx, ly, u, rho, g):
    # Zou-He no-slip top right corner velocity b.c.
    u[0, lx, ly] = u[0, lx - 1, ly]
    u[1, lx, ly] = u[1, lx - 1, ly]

    rho[lx, ly] = rho[lx - 1, ly]

    g[2, lx, ly] = (g[1, lx, ly] - (2.0 / 3.0) * rho[lx, ly] * u[0, lx, ly])

    g[4, lx, ly] = (g[3, lx, ly] - (2.0 / 3.0) * rho[lx, ly] * u[1, lx, ly])

    g[6, lx, ly] = (g[5, lx, ly] - (1.0 / 6.0) * rho[lx, ly] *
                    u[0, lx, ly] - (1.0 / 6.0) * rho[lx, ly] * u[1, lx, ly])

    g[7, lx, ly] = 0.0
    g[8, lx, ly] = 0.0

    g[0, lx, ly] = (rho[lx, ly]
                    - g[1, lx, ly] - g[2, lx, ly] - g[3, lx, ly] - g[4, lx, ly]
                    - g[5, lx, ly] - g[6, lx, ly] - g[7, lx, ly] - g[8, lx, ly])


@jit(nopython=True, cache=True)
def nb_zou_he_bottom_right_corner_velocity(lx, ly, u, rho, g):
    # Zou-He no-slip bottom right corner velocity b.c.
    u[0, lx, 0] = u[0, lx - 1, 0]
    u[1, lx, 0] = u[1, lx - 1, 0]

    rho[lx, 0] = rho[lx - 1, 0]

    g[2, lx, 0] = (g[1, lx, 0] - (2.0 / 3.0) * rho[lx, 0] * u[0, lx, 0])
    g[3, lx, 0] = (g[4, lx, 0] + (2.0 / 3.0) * rho[lx, 0] * u[1, lx, 0])
    g[7, lx, 0] = (g[8, lx, 0] - (1.0 / 6.0) * rho[lx, 0] *
                   u[0, lx, 0] + (1.0 / 6.0) * rho[lx, 0] * u[1, lx, 0])

    g[5, lx, 0] = 0.0
    g[6, lx, 0] = 0.0

    g[0, lx, 0] = (rho[lx, 0] - g[1, lx, 0] - g[2, lx, 0] - g[3, lx, 0] -
                   g[4, lx, 0] - g[5, lx, 0] - g[6, lx, 0] - g[7, lx, 0] - g[8, lx, 0])
