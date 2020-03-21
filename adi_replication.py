
import numpy as np
from scipy.interpolate import interp2d
from amc.helper import get_european_call_bs
import time


def build_a11(r: float, q1: float, sig1: float, q2: float, sig2: float, xs: np.ndarray, ys: np.ndarray):
    m = len(xs)
    n = len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    mat = np.zeros((m * n, m * n))

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            mat[i * n + j, (i - 1) * n + j] = (r - q1) * xs[i] / 2 / dx
            mat[i * n + j, (i + 1) * n + j] = -(r - q1) * xs[i] / 2 / dx

    return mat


def build_a12(r: float, q1: float, sig1: float, q2: float, sig2: float, xs: np.ndarray, ys: np.ndarray):
    m = len(xs)
    n = len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    mat = np.zeros((m * n, m * n))

    const = 0.5 * sig1 * sig1 / dx / dx
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            mat[i * n + j, (i - 1) * n + j] = const * xs[i] * xs[i]
            mat[i * n + j, i * n + j] = -2 * const * xs[i] * xs[i] - 0.5 * r
            mat[i * n + j, (i + 1) * n + j] = const * xs[i] * xs[i]

    return mat


def build_a21(r: float, q1: float, sig1: float, q2: float, sig2: float, xs: np.ndarray, ys: np.ndarray):
    m = len(xs)
    n = len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    mat = np.zeros((m * n, m * n))

    for i in range(1, m - 1):
        for j in range(1, n - 1):
            mat[i * n + j, i * n + j - 1] = (r - q2) * ys[j] / 2 / dy
            mat[i * n + j, i * n + j + 1] = -(r - q2) * ys[j] / 2 / dy

    return mat


def build_a22(r: float, q1: float, sig1: float, q2: float, sig2: float, xs: np.ndarray, ys: np.ndarray):
    m = len(xs)
    n = len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    mat = np.zeros((m * n, m * n))

    const = 0.5 * sig2 * sig2 / dy / dy
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            mat[i * n + j, i * n + j - 1] = const * ys[j] * ys[j]
            mat[i * n + j, i * n + j] = -2 * const * ys[j] * ys[j] - 0.5 * r
            mat[i * n + j, i * n + j + 1] = const * ys[j] * ys[j]

    return mat


def build_a_mixed(rho: float, r: float, q1: float, sig1: float, q2: float, sig2: float, xs: np.ndarray, ys: np.ndarray):
    m = len(xs)
    n = len(ys)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    mat = np.zeros((m * n, m * n))

    const = rho * sig1 * sig2 / 4 / dx / dy
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            mat[i * n + j, (i - 1) * n + j - 1] = const * xs[i] * ys[j]
            mat[i * n + j, (i - 1) * n + j + 1] = -const * xs[i] * ys[j]
            mat[i * n + j, (i + 1) * n + j - 1] = -const * xs[i] * ys[j]
            mat[i * n + j, (i + 1) * n + j + 1] = const * xs[i] * ys[j]

    return mat


def build_boundaries(r: float, q1: float, sig1: float, q2: float, sig2: float, xs: np.ndarray, ys: np.ndarray):
    mat = np.zeros(m * n)

    for i in range(m):
        mat[i * n + 0] = max(xs[i] - ys[0], 0)
        mat[(i + 1) * n - 1] = max(xs[i] - ys[n - 1], 0)

    for j in range(n):
        mat[0 + j] = max(xs[0] - ys[j], 0)
        mat[(m - 1) * n + j] = max(xs[m - 1] - ys[j], 0)
    return mat


def step(values, dt, a0, a1, a2):
    I = np.eye(len(values))
    A = a0 + a1 + a2
    y0 = (I + dt * A) @ values
    y1 = np.linalg.solve(I - 0.5 * dt * a1, y0 - (0.5 * dt * a1) @ values)
    y2 = np.linalg.solve(I - 0.5 * dt * a2, y1 - (0.5 * dt * a2) @ values)
    return y2


def hv_step(values, dt, a0, a1, a2):
    I = np.eye(len(values))
    A = a0 + a1 + a2
    y0 = (I + dt * A) @ values
    y1 = np.linalg.solve(I - 0.5 * dt * a1, y0 - (0.5 * dt * a1) @ values)
    y2 = np.linalg.solve(I - 0.5 * dt * a2, y1 - (0.5 * dt * a2) @ values)
    y3 = y0 + (0.5 * dt * A) @ (y2 - values)
    y4 = np.linalg.solve(I - 0.5 * dt * a1, y3 - (0.5 * dt * a1) @ y2)
    y5 = np.linalg.solve(I - 0.5 * dt * a2, y4 - (0.5 * dt * a2) @ y2)
    return y5


def cs_step(values, dt, a0, a1, a2):
    I = np.eye(len(values))
    A = a0 + a1 + a2
    y0 = (I + dt * A) @ values
    y1 = np.linalg.solve(I - 0.5 * dt * a1, y0 - (0.5 * dt * a1) @ values)
    y2 = np.linalg.solve(I - 0.5 * dt * a2, y1 - (0.5 * dt * a2) @ values)
    y3 = y0 + (0.5 * dt * a0) @ (y2 - values)
    y4 = np.linalg.solve(I - 0.5 * dt * a1, y3 - (0.5 * dt * a1) @ values)
    y5 = np.linalg.solve(I - 0.5 * dt * a2, y4 - (0.5 * dt * a2) @ values)
    return y5


def mcs_step(values, dt, a0, a1, a2):
    I = np.eye(len(values))
    A = a0 + a1 + a2
    a0, a1, a2 = 0.5 * dt * a0, 0.5 * dt * a1, 0.5 * dt * a2
    y0 = (I + dt * A) @ values
    y1 = np.linalg.solve(I - a1, y0 - a1 @ values)
    y2 = np.linalg.solve(I - a2, y1 - a2 @ values)
    y3 = y0 + (a0 + 0.5 * dt * A) @ (y2 - values)
    y4 = np.linalg.solve(I - a1, y3 - a1 @ values)
    y5 = np.linalg.solve(I - a2, y4 - a2 @ values)
    return y5


if __name__ == '__main__':
    s1, q1, sig1 = 90, 0, 0.3
    s2, q2, sig2 = 110, 0, 0.4
    r = 0.01
    t = 0.4
    rho = 0.4

    m = 50
    n = 50
    std = 5
    n_steps = 20
    dt = t / n_steps

    xs = np.linspace(s1 / np.exp(std * sig1 * np.sqrt(t)), s1 * np.exp(std * sig1 * np.sqrt(t)), m)
    ys = np.linspace(s2 / np.exp(std * sig2 * np.sqrt(t)), s2 * np.exp(std * sig2 * np.sqrt(t)), n)

    curr = np.zeros(m * n)
    prev = np.maximum(xs.reshape(-1, 1) - ys, 0).flatten()
    a11 = build_a11(r, q1, sig1, q2, sig2, xs, ys)
    a12 = build_a12(r, q1, sig1, q2, sig2, xs, ys)
    a1 = a11 + a12

    a21 = build_a21(r, q1, sig1, q2, sig2, xs, ys)
    a22 = build_a22(r, q1, sig1, q2, sig2, xs, ys)
    a2 = a21 + a22

    a0 = build_a_mixed(rho, r, q1, sig1, q2, sig2, xs, ys)
    bounds = build_boundaries(r, q1, sig1, q2, sig2, xs, ys)

    start = time.time()
    for i in range(n_steps):
        curr = step(prev, dt, a0, a1, a2)
        prev = curr
        print(i)
    curr = curr.reshape(m, n)
    print(f'{time.time() - start:.2f}s')

    f = interp2d(ys, xs, curr)
    diff = []

    for x in np.linspace(s1 / np.exp(2 * sig1 * np.sqrt(t)), s1 * np.exp(2 * sig1 * np.sqrt(t)), 10):
        for y in np.linspace(s2 / np.exp(2 * sig2 * np.sqrt(t)), s2 * np.exp(2 * sig2 * np.sqrt(t)), 10):
            ans = f(y, x)[0]
            expected = get_european_call_bs(x, y, 0, 0, np.sqrt(sig1 ** 2 + sig2 ** 2 - 2 * rho * sig1 * sig2), t)
            diff.append(ans - expected)

    diff = np.array(diff)
    # print(diff)
    print(np.sqrt(np.sum(diff ** 2)))
