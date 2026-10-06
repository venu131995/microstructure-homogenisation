"""Periodic finite-element homogenisation of effective thermal conductivity (bilinear Q1 elements).

For a periodic unit cell Y made of n-by-n square pixels with conductivity k_e, the effective tensor is

    K_ij = (1/|Y|) sum_e (chi0_i - chi_i)_e^T  k_e KE0  (chi0_j - chi_j)_e

where chi0_i is the element field of a unit macroscopic gradient in direction i and chi_i is the
periodic fluctuation solving  K chi_i = F_i  (Andreassen & Andreasen 2014). Periodicity is imposed by
identifying opposite boundary nodes, i.e. the mesh is a torus with n*n nodes; one node is fixed to
remove the constant null space. Both load cases share one sparse factorisation.

The tensor is self-adjoint in the design, so its sensitivity needs no extra solve:
    dK_ij/drho_e = (1/|Y|) dk_e/drho_e (chi0_i - chi_i)_e^T KE0 (chi0_j - chi_j)_e
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Q1 element on a unit square, nodes ordered (0,0), (1,0), (1,1), (0,1)
KE0 = np.array([[4, -1, -2, -1], [-1, 4, -1, -2], [-2, -1, 4, -1], [-1, -2, -1, 4]], float) / 6.0
CHI0 = np.array([[0.0, 1.0, 1.0, 0.0],       # unit gradient in x: nodal values of x
                 [0.0, 0.0, 1.0, 1.0]])      # unit gradient in y: nodal values of y


@dataclass
class Mesh:
    n: int
    edof: np.ndarray                 # (n*n, 4) periodic node indices of each element

    @classmethod
    def periodic(cls, n: int) -> "Mesh":
        j, i = np.divmod(np.arange(n * n), n)               # element (row j, column i)
        node = lambda ii, jj: (jj % n) * n + (ii % n)         # noqa: E731
        edof = np.stack([node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)], axis=1)
        return cls(n=n, edof=edof)


def homogenise(k: np.ndarray, mesh: Mesh, return_fields: bool = False):
    """Effective 2x2 conductivity tensor and the element strain-energy terms for sensitivities.

    Returns K (2x2) and E (n*n, 2, 2) with E[e, i, j] = (chi0_i - chi_i)_e^T KE0 (chi0_j - chi_j)_e,
    so that K = sum_e k_e E[e] / n^2. With return_fields=True also returns the element nodal values
    of the total field (chi0 - chi), shape (n*n, 4, 2), for post-processing heat fluxes.
    """
    n2 = mesh.n * mesh.n
    rows = np.repeat(mesh.edof, 4, axis=1).ravel()
    cols = np.tile(mesh.edof, (1, 4)).ravel()
    vals = (k[:, None, None] * KE0[None]).reshape(n2, 16).ravel()
    A = sp.csc_matrix((vals, (rows, cols)), shape=(n2, n2))
    F = np.zeros((n2, 2))
    fe = k[:, None, None] * (KE0 @ CHI0.T)[None]            # (n2, 4, 2)
    for c in range(2):
        np.add.at(F[:, c], mesh.edof.ravel(), fe[:, :, c].ravel())
    free = np.arange(1, n2)                                 # fix node 0
    chi = np.zeros((n2, 2))
    lu = spla.splu(A[free][:, free])
    chi[free] = lu.solve(F[free])
    d = CHI0.T[None, :, :] - chi[mesh.edof]                 # (n2, 4, 2): (chi0 - chi) per element
    E = np.einsum("eai,ab,ebj->eij", d, KE0, d)
    K = np.einsum("e,eij->ij", k, E) / n2
    if return_fields:
        return K, E, d
    return K, E


def element_flux(k: np.ndarray, d: np.ndarray, load_case: int = 0) -> np.ndarray:
    """Heat flux vector at each element centre for a unit macroscopic gradient (Q1 on unit squares)."""
    v = d[:, :, load_case]
    gx = 0.5 * ((v[:, 1] - v[:, 0]) + (v[:, 2] - v[:, 3]))
    gy = 0.5 * ((v[:, 3] - v[:, 0]) + (v[:, 2] - v[:, 1]))
    return -k[:, None] * np.stack([gx, gy], axis=1)


# ----------------------------------------------------------------------------- analytical references

def voigt_reuss(k1: float, k2: float, f1: float) -> tuple[float, float]:
    return f1 * k1 + (1 - f1) * k2, 1 / (f1 / k1 + (1 - f1) / k2)


def hashin_shtrikman_2d(k_hi: float, k_lo: float, f_hi: float) -> tuple[float, float]:
    """2-D Hashin-Shtrikman bounds for an isotropic two-phase composite (lower, upper)."""
    f_lo = 1 - f_hi
    upper = k_hi + f_lo / (1 / (k_lo - k_hi) + f_hi / (2 * k_hi))
    lower = k_lo + f_hi / (1 / (k_hi - k_lo) + f_lo / (2 * k_lo))
    return lower, upper


def maxwell_garnett_2d(k_inc: float, k_mat: float, f_inc: float) -> float:
    """Maxwell-Garnett / Rayleigh dilute estimate for circular inclusions (= HS bound with matrix as reference)."""
    beta = (k_inc - k_mat) / (k_inc + k_mat)
    return k_mat * (1 + f_inc * beta) / (1 - f_inc * beta)


# ----------------------------------------------------------------------------- geometries

def circle_cell(n: int, f: float) -> np.ndarray:
    """Pixel indicator (1 = inclusion) of a centred circle with area fraction ~f (sub-pixel averaged)."""
    r = np.sqrt(f / np.pi)
    sub = 4
    s = (np.arange(n * sub) + 0.5) / (n * sub) - 0.5
    X, Y = np.meshgrid(s, s)
    inside = (X**2 + Y**2 <= r**2).astype(float)
    return inside.reshape(n, sub, n, sub).mean(axis=(1, 3)).ravel()


def checkerboard(n: int, squares: int = 2) -> np.ndarray:
    j, i = np.divmod(np.arange(n * n), n)
    return (((i * squares // n) + (j * squares // n)) % 2).astype(float)


def laminate(n: int, f: float, axis: str = "x") -> np.ndarray:
    j, i = np.divmod(np.arange(n * n), n)
    coord = j if axis == "x" else i                         # layers parallel to the x axis -> vary in y
    return (coord < round(f * n)).astype(float)
