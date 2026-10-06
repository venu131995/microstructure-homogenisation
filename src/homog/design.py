"""Inverse homogenisation: design a periodic unit cell for target effective conductivity.

Maximise the trace of the effective tensor (k_xx + k_yy) at a fixed volume fraction of the
conducting phase. In 2-D the Hashin-Shtrikman upper bound limits the trace to 2 k_HS for any
microstructure, so the attained fraction of the bound measures how good the design is.

Ingredients: SIMP interpolation k = k_lo + (k_hi - k_lo) rho^p, a *periodic* density filter,
Heaviside projection with continuation, eight-fold (D4) symmetry so designs are square-symmetric
(hence isotropic in 2-D), and MMA. Sensitivities come from the self-adjoint formula in fem.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .fem import Mesh, hashin_shtrikman_2d, homogenise


def periodic_filter(n: int, radius: float) -> sp.csr_matrix:
    r = int(np.ceil(radius))
    rows, cols, vals = [], [], []
    for j in range(n):
        for i in range(n):
            for dj in range(-r, r + 1):
                for di in range(-r, r + 1):
                    w = radius - np.hypot(di, dj)
                    if w > 0:
                        rows.append(j * n + i); cols.append(((j + dj) % n) * n + (i + di) % n); vals.append(w)
    H = sp.csr_matrix((vals, (rows, cols)), shape=(n * n, n * n))
    return sp.diags(1 / np.asarray(H.sum(axis=1)).ravel()) @ H


def project(x, beta, eta=0.5):
    tb = np.tanh(beta * eta)
    den = tb + np.tanh(beta * (1 - eta))
    return (tb + np.tanh(beta * (x - eta))) / den, beta * (1 - np.tanh(beta * (x - eta)) ** 2) / den


def symmetrise(v: np.ndarray, n: int) -> np.ndarray:
    """Average over the 8 symmetries of the square (an orthogonal projection, so it is self-adjoint)."""
    a = v.reshape(n, n)
    ops = [a, a.T, a[::-1], a[:, ::-1], a[::-1].T, a[:, ::-1].T, a[::-1, ::-1], a[::-1, ::-1].T]
    return np.mean(ops, axis=0).ravel()


class MMA:
    """Svanberg MMA with one exact linear constraint (see Project 1 for the derivation)."""

    def __init__(self, move: float = 0.2):
        self.move, self.it, self.x1, self.x2, self.low, self.upp = move, 0, None, None, None, None

    def step(self, x, df, a, b):
        self.it += 1
        if self.it <= 2:
            self.low, self.upp = x - 0.5, x + 0.5
        else:
            sgn = (x - self.x1) * (self.x1 - self.x2)
            gam = np.where(sgn > 0, 1.2, np.where(sgn < 0, 0.7, 1.0))
            self.low = np.clip(x - gam * (self.x1 - self.low), x - 10, x - 0.01)
            self.upp = np.clip(x + gam * (self.upp - self.x1), x + 0.01, x + 10)
        lo = np.maximum.reduce([np.zeros_like(x), self.low + 0.1 * (x - self.low), x - self.move])
        hi = np.minimum.reduce([np.ones_like(x), self.upp - 0.1 * (self.upp - x), x + self.move])
        dfp, dfm = np.maximum(df, 0), np.maximum(-df, 0)
        reg = 1e-3 * np.abs(df) + 1e-6 / (self.upp - self.low)
        p = (self.upp - x) ** 2 * (1.001 * dfp + 0.001 * dfm + reg)
        q = (x - self.low) ** 2 * (0.001 * dfp + 1.001 * dfm + reg)

        def x_of(lam):
            l, h = lo.copy(), hi.copy()
            for _ in range(60):
                m = 0.5 * (l + h)
                pos = p / (self.upp - m) ** 2 - q / (m - self.low) ** 2 + lam * a > 0
                h, l = np.where(pos, m, h), np.where(pos, l, m)
            return 0.5 * (l + h)

        if a @ x_of(0.0) <= b:
            xn = x_of(0.0)
        else:
            l, h = 0.0, 1.0
            doublings = 0
            while a @ x_of(h) > b and doublings < 60:     # unreachable target within move limits:
                h *= 2                                     # take the most volume-reducing feasible step
                doublings += 1
            for _ in range(60):
                m = 0.5 * (l + h)
                l, h = (m, h) if a @ x_of(m) > b else (l, m)
            xn = x_of(h)
        self.x2, self.x1 = (self.x1 if self.x1 is not None else x.copy()), x.copy()
        return xn


@dataclass
class DesignResult:
    rho: np.ndarray
    K: np.ndarray
    hs_upper: float
    history: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)      # (iteration, physical density)

    @property
    def bound_fraction(self) -> float:
        return float(np.trace(self.K) / (2 * self.hs_upper))


def maximise_conductivity(n: int, vol: float, k_hi: float = 1.0, k_lo: float = 1e-3, radius: float = 2.0,
                          iters: int = 300, betas: tuple = (1, 2, 4, 8, 16, 32), every: int = 50, penal: float = 3.0,
                          seed: int = 0, snapshot_every: int = 0) -> DesignResult:
    mesh, H = Mesh.periodic(n), periodic_filter(n, radius)
    # start from a centred circular hole (standard symmetry-breaking initial guess, Sigmund 1994);
    # a uniform grey start is a stationary point of the trace objective under SIMP
    j, i = np.divmod(np.arange(n * n), n)
    r = np.hypot(i + 0.5 - n / 2, j + 0.5 - n / 2) / n
    x = np.where(r < 0.25, 0.0, 1.0)
    x = np.clip(x * vol / x.mean(), 0, 1)
    x = symmetrise(x + 0.02 * np.random.default_rng(seed).standard_normal(n * n), n).clip(0, 1)
    mma, hist, snaps = MMA(), [], []
    for it in range(iters):
        beta = betas[min(it // every, len(betas) - 1)]
        rho, drho = project(H @ x, beta)
        k = k_lo + (k_hi - k_lo) * rho**penal
        K, E = homogenise(k, mesh)
        dk = penal * (k_hi - k_lo) * rho ** (penal - 1)
        dtrace = dk * (E[:, 0, 0] + E[:, 1, 1]) / (n * n)
        df = -symmetrise(H.T @ (dtrace * drho), n) / np.trace(K)              # minimise -trace (scaled)
        a = symmetrise(H.T @ (drho / (n * n)), n)
        b = vol - rho.mean() + a @ x
        x = symmetrise(mma.step(x, df, a, b), n)
        hist.append({"it": it, "beta": beta, "trace": float(np.trace(K)), "vol": float(rho.mean())})
        if snapshot_every and it % snapshot_every == 0:
            snaps.append((it, rho.copy()))
    rho, _ = project(H @ x, betas[-1])
    crisp = (rho > 0.5).astype(float)
    K, _ = homogenise(k_lo + (k_hi - k_lo) * crisp, mesh)
    return DesignResult(rho=crisp, K=K, hs_upper=hashin_shtrikman_2d(k_hi, k_lo, crisp.mean())[1], history=hist,
                        snapshots=snaps)
