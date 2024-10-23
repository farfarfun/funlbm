import numpy as np

cache_map = {}


def cul_u(y, z, a, b, size=100):
    n = np.arange(size)
    res = 32 * (-1) ** (n + 1) / (2 * n + 1) ** 3 / np.pi**3
    res = res * np.cos((2 * n + 1) / 2 / a * np.pi * y)
    res = res * np.cosh((2 * n + 1) / 2 / a * np.pi * z) / np.cosh((2 * n + 1) / 2 / a * np.pi * b)

    res = 1 - y**2 / a**2 + np.sum(res)
    return round(res, 6)


def init_u(a, b, u_max=0.01, n_max=100):
    key = f"u{a}-{b}"
    if key in cache_map.keys():
        return cache_map[key]

    res = np.zeros([a, b])
    for i in range(a):
        yi = i - (a - 1) / 2.0
        for j in range(b):
            zi = j - (b - 1) / 2.0
            res[i, j] = u_max * cul_u(yi, zi, a / 2.0, b / 2.0, size=n_max)
    cache_map[key] = res
    return res
