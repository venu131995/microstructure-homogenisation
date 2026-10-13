"""Periodic Stokes cell problem: effective permeability of a two-phase pixel microstructure.

Unit cell Y = [0,1]^2 on an n-by-n pixel grid; solid pixels are blocked with a Brinkman penalty
(alpha -> large), fluid pixels are free. For a unit body force e_i (a unit mean pressure gradient):

    -mu lap(w) + alpha(x) w + grad(pi) = e_i ,   div(w) = 0 ,   w, pi periodic

and the permeability tensor is K_ij = <w_i . e_j>, the cell-averaged (superficial) velocity, so that
Darcy's law <u> = -(K/mu) grad<p> holds at the device scale. Lengths are in units of the cell size L;
multiply K by L^2 for dimensional permeability.

Discretisation: staggered MAC grid (u on vertical faces, v on horizontal faces, p at centres), second-order
central differences, fully periodic. The pressure gradient is the exact negative transpose of the discrete
divergence, so the discrete power identity  <f.u> = mu |grad u|^2 + alpha |u|^2  holds to round-off.
One continuity row is replaced by p_0 = 0 (the sum of all continuity rows vanishes identically).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _shift(n, di, dj):
    """Sparse operator returning the value at (i+di, j+dj), periodic; unknown id = j*n + i."""
    j, i = np.divmod(np.arange(n * n), n)
    cols = ((j + dj) % n) * n + (i + di) % n
    return sp.csr_matrix((np.ones(n * n), (np.arange(n * n), cols)), shape=(n * n, n * n))


def face_masks(n: int, inside):
    """Solid indicator evaluated exactly at u-face centres (i h, (j+1/2) h) and v-face centres ((i+1/2) h, j h).
    inside(x, y) -> bool array, periodic geometry on the unit cell."""
    j, i = np.divmod(np.arange(n * n), n)
    h = 1.0 / n
    return inside(i * h, (j + 0.5) * h).astype(float), inside((i + 0.5) * h, j * h).astype(float)


def assemble(solid: np.ndarray, mu: float = 1.0, alpha_max: float = 1e8, faces=None):
    """solid: n*n array in [0,1] (1 = blocked pixel). Returns the saddle-point matrix and helpers.
    faces: optional (s_u, s_v) solid indicators at the velocity points (from face_masks); otherwise they are
    averaged from the two pixels sharing each face."""
    n = int(round(np.sqrt(solid.size)))
    h = 1.0 / n
    I = sp.identity(n * n, format="csr")
    E, W, N, S = _shift(n, 1, 0), _shift(n, -1, 0), _shift(n, 0, 1), _shift(n, 0, -1)
    lap = (E + W + N + S - 4 * I) / h**2
    # face solid fractions: u-face (i,j) sits between cells (i-1,j) and (i,j); v-face between (i,j-1) and (i,j)
    if faces is not None:
        s_u, s_v = faces
    else:
        s_u = 0.5 * (solid + W @ solid)
        s_v = 0.5 * (solid + S @ solid)
    Au = -mu * lap + sp.diags(alpha_max * s_u)
    Av = -mu * lap + sp.diags(alpha_max * s_v)
    Dx = (E - I) / h          # (div u)_cell = (u_{i+1} - u_i)/h  with u_i on the left face of cell i
    Dy = (N - I) / h
    Gx, Gy = -Dx.T, -Dy.T     # grad p on faces = -(div)^T  -> exact discrete energy identity
    Z = sp.csr_matrix((n * n, n * n))
    A = sp.bmat([[Au, Z, Gx], [Z, Av, Gy], [Dx, Dy, None]], format="lil")
    m = 2 * n * n                                   # first continuity row -> p_0 = 0
    A[m, :] = 0
    A[m, m] = 1.0
    return A.tocsc(), dict(n=n, h=h, Au=Au, Av=Av, Dx=Dx, Dy=Dy, s_u=s_u, s_v=s_v, mu=mu, alpha=alpha_max)


def permeability(solid: np.ndarray, mu: float = 1.0, alpha_max: float = 1e8, return_fields: bool = False,
                 faces=None):
    """2x2 permeability tensor (cell-size units) for a pixel microstructure; solid = 1 blocked."""
    A, H = assemble(np.asarray(solid, float), mu, alpha_max, faces)
    n2 = H["n"] ** 2
    lu = spla.splu(A)
    K = np.zeros((2, 2))
    fields = []
    for c in range(2):
        b = np.zeros(3 * n2)
        b[c * n2:(c + 1) * n2] = 1.0                 # unit body force in direction c
        b[2 * n2] = 0.0
        x = lu.solve(b)
        u, v, p = x[:n2], x[n2:2 * n2], x[2 * n2:]
        K[c, 0], K[c, 1] = u.mean(), v.mean()
        fields.append((u, v, p))
    if return_fields:
        return K, fields, H
    return K


def balances(solid: np.ndarray, mu: float = 1.0, alpha_max: float = 1e8, faces=None) -> dict:
    """Mass and power balances of the x load case."""
    K, fields, H = permeability(solid, mu, alpha_max, return_fields=True, faces=faces)
    u, v, p = fields[0]
    div = H["Dx"] @ u + H["Dy"] @ v
    power_in = u.mean()                                      # <f . u> with f = e_x
    dissip = float(u @ (H["Au"] @ u) + v @ (H["Av"] @ v)) / H["n"] ** 2
    umax = max(np.abs(u).max(), 1e-300)
    return {"mass_max_divergence_rel": float(np.abs(div).max() * H["h"] / umax),
            "power_in": float(power_in), "dissipation": dissip,
            "power_closure_rel": float((dissip - power_in) / power_in) if power_in else 0.0,
            "K_symmetry": float(abs(K[0, 1] - K[1, 0]) / max(abs(K).max(), 1e-300))}


# ----------------------------------------------------------------------------- analytical references

def poiseuille_layers(fluid_fraction: float) -> float:
    """Flow along fluid layers of thickness phi (cell units) between solid layers: K = phi^3 / 12."""
    return fluid_fraction**3 / 12.0


def cylinder_inside(solid_fraction: float):
    """Periodic centred cylinder of the given area fraction: inside(x, y) for face_masks."""
    r2 = solid_fraction / np.pi

    def inside(x, y):
        dx, dy = np.mod(x, 1.0) - 0.5, np.mod(y, 1.0) - 0.5
        return dx * dx + dy * dy < r2
    return inside


def drummond_tahir(solid_fraction: float) -> float:
    """Square array of cylinders, flow perpendicular to the axes (Drummond & Tahir 1984), K in cell units^2."""
    c = solid_fraction
    a2 = c / np.pi                                            # cylinder radius^2 in a unit cell
    return a2 / (8 * c) * (-np.log(c) - 1.476 + 2 * c - 1.774 * c**2 + 4.076 * c**3)


# ----------------------------------------------------------------------------- finite channel (Project 2c check)

def _shift2(nx, ny, di, dj):
    j, i = np.divmod(np.arange(nx * ny), nx)
    cols = ((j + dj) % ny) * nx + (i + di) % nx
    return sp.csr_matrix((np.ones(nx * ny), (np.arange(nx * ny), cols)), shape=(nx * ny, nx * ny))


def channel_flow(n: int, inside, n_rows: int, mu: float = 1.0, alpha_max: float = 1e8) -> dict:
    """Fully resolved flow through n_rows pin cells stacked across a channel bounded by two no-slip walls.

    Domain: one cell along x (periodic: the array is long), n_rows cells across plus one blocked pixel row that
    acts as the top and bottom wall (periodic wrap in y). Unit body force along x. Returns the resolved flow rate
    per unit pressure gradient and the Darcy prediction K * n_rows from the periodic cell."""
    nx, ny = n, n * n_rows + 1
    h = 1.0 / n
    j, i = np.divmod(np.arange(nx * ny), nx)
    wall = j == ny - 1
    yu, yv = (j + 0.5) * h, j * h
    s_u = (inside(i * h, yu) | wall).astype(float)
    s_v = (inside((i + 0.5) * h, yv) | wall | (j == 0)).astype(float)
    I = sp.identity(nx * ny, format="csr")
    E, W, N, S = _shift2(nx, ny, 1, 0), _shift2(nx, ny, -1, 0), _shift2(nx, ny, 0, 1), _shift2(nx, ny, 0, -1)
    lap = (E + W + N + S - 4 * I) / h**2
    Au = -mu * lap + sp.diags(alpha_max * s_u)
    Av = -mu * lap + sp.diags(alpha_max * s_v)
    Dx, Dy = (E - I) / h, (N - I) / h
    Z = sp.csr_matrix((nx * ny, nx * ny))
    A = sp.bmat([[Au, Z, -Dx.T], [Z, Av, -Dy.T], [Dx, Dy, None]], format="lil")
    m = 2 * nx * ny
    A[m, :] = 0
    A[m, m] = 1.0
    b = np.zeros(3 * nx * ny)
    b[:nx * ny] = 1.0
    x = spla.spsolve(A.tocsc(), b)
    u, v = x[:nx * ny], x[nx * ny:2 * nx * ny]
    Q = u.reshape(ny, nx).sum(axis=0).mean() * h               # flow rate through a cross-section
    div = Dx @ u + Dy @ v
    K = permeability(np.zeros(n * n), faces=face_masks(n, inside))[0, 0]
    return {"n_rows": n_rows, "Q_resolved": float(Q), "Q_darcy": float(K * n_rows),
            "ratio": float(Q / (K * n_rows)), "u": u.reshape(ny, nx),
            "mass_max_divergence_rel": float(np.abs(div).max() * h / max(np.abs(u).max(), 1e-300))}
