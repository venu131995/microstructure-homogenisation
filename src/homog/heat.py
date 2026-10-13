"""Convective heat transfer in periodic copper/water cells (Project 2c) and the fully resolved device check.

All quantities are dimensionless: cell size L = 1, water conductivity k_f = 1, water heat capacity (rho c)_f = 1,
so the Darcy (superficial) velocity equals the cell Peclet number Pe = (rho c)_f U_D L / k_f.
Copper heat capacity never enters: every problem here is steady.

Discretisation: cell-centred finite volumes on the pixel grid of the Stokes solver (stokes.py), velocities on the
MAC faces, harmonic-mean face conductivities, central advection (conservative, exact for linear fields).
Unknown id = j*nx + i (row j, column i).

Cell problems (periodic unit cell, Stokes velocity scaled to mean U_D = Pe):

  h_v  : uniform heat q in the copper, periodically developed temperature T = gamma x + theta(x),
         gamma = q phi_s / U_D from the energy balance. Volumetric interfacial coefficient
         h_v = q phi_s / (<T>_s - T_ref), with T_ref the bulk (velocity-weighted) or the plain mean water temperature.
  k_f,eff : heat spreading along x inside the water alone (copper insulating), including the extra spreading
         by the non-uniform velocity (Taylor dispersion). Closure  u.grad b - div(k grad b) = -(u_x - U) + div(k e_x),
         K = <k (1 + db/dx)>_f - <(u_x - U) b>_f  (intrinsic), k_f,eff = phi_f K (superficial).

Device model (two-temperature, copper pins disconnected so copper conduction in the plane is zero):

  water : U_D dT_b/dx = k_f,eff d2T_b/dx2 + phi_s q(x)      T_b(0) = 0,  dT_b/dx(L) = 0
  copper: T_s = T_b + phi_s q(x) / h_v
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .stokes import face_masks, permeability


# ----------------------------------------------------------------------------- finite-volume assembly

def _harm(a, b):
    s = a + b
    out = np.zeros_like(s)
    m = s > 0
    out[m] = 2 * a[m] * b[m] / s[m]
    return out


def assemble(k, u, v, h, periodic_x=True, active=None):
    """Steady advection-diffusion operator  div(u T) - div(k grad T)  on an ny-by-nx grid (periodic in y).

    k: (ny, nx) cell conductivity; u: (ny, nx+1) x-velocity on vertical faces (u[:, i] = left face of column i);
    v: (ny, nx) y-velocity on bottom faces (periodic). Non-periodic x: Dirichlet inlet at x = 0 (value supplied
    through inlet_rhs), outflow (upwind, no conduction) at x = nx*h. active: optional bool mask; inactive cells
    get T = 0 rows and no coupling. Returns (A, inlet_rhs, faces) with faces = dict of face conductivities.
    """
    ny, nx = k.shape
    idx = np.arange(nx * ny).reshape(ny, nx)
    rows, cols, vals = [], [], []

    def add(r, c, x):
        rows.append(r.ravel()); cols.append(c.ravel()); vals.append(np.broadcast_to(x, r.shape).ravel())

    # x-faces between column i-1 and i, i = 1..nx-1 (plus the periodic face i = 0)
    if periodic_x:
        kL, kR = np.roll(k, 1, axis=1), k
        kx = _harm(kL, kR)                                  # kx[:, i] = face on the left of column i
        PL, PR = np.roll(idx, 1, axis=1), idx
        F = u[:, :nx]
        sel = (slice(None), slice(None))
    else:
        kx = np.zeros((ny, nx + 1))
        kx[:, 1:nx] = _harm(k[:, :-1], k[:, 1:])
        PL, PR = idx[:, :-1], idx[:, 1:]
        F = u[:, 1:nx]
        sel = (slice(None), slice(1, nx))
    G = kx[sel] / h**2
    A_ = F / (2 * h)
    # left cell PL: outflow +F through its east face; right cell PR: inflow through its west face
    add(PL, PL, G + A_); add(PL, PR, -G + A_)
    add(PR, PR, G - A_); add(PR, PL, -G - A_)
    # y-faces (always periodic): face on the bottom of row j
    ky = _harm(np.roll(k, 1, axis=0), k)
    PB, PT = np.roll(idx, 1, axis=0), idx
    Gy, Ay = ky / h**2, v / (2 * h)
    add(PB, PB, Gy + Ay); add(PB, PT, -Gy + Ay)
    add(PT, PT, Gy - Ay); add(PT, PB, -Gy - Ay)
    inlet_rhs = np.zeros((ny, nx))
    if not periodic_x:
        kx[:, 0] = k[:, 0]                                  # inlet face: wall value at half-cell distance
        c0 = idx[:, 0]
        add(c0, c0, 2 * k[:, 0] / h**2)                     # conduction to the Dirichlet inlet value
        inlet_rhs[:, 0] = 2 * k[:, 0] / h**2 + u[:, 0] / h  # times T_in: conduction + advected inflow
        cN = idx[:, -1]
        add(cN, cN, u[:, nx] / h)                           # upwind outflow, zero conductive flux
        kx[:, nx] = 0.0
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nx * ny, nx * ny))
    if active is not None:
        off = ~active.ravel()
        D = sp.diags((~off).astype(float))
        A = (D @ A + sp.diags(off.astype(float))).tocsr()
    return A, inlet_rhs, {"kx": kx, "ky": ky}


def _pin_solve(A, rhs, cell):
    """Solve a singular (constant null space) periodic system by fixing one unknown to zero."""
    A = A.tolil()
    A[cell, :] = 0
    A[cell, cell] = 1.0
    rhs = rhs.copy()
    rhs[cell] = 0.0
    return spla.spsolve(A.tocsc(), rhs)


# ----------------------------------------------------------------------------- unit-cell flow

def cell_flow(n, inside, Pe=1.0):
    """Stokes flow through the periodic unit cell, scaled to Darcy velocity U_D = Pe along x.
    Returns u (n, n+1), v (n, n), solid pixel mask (n, n), permeability K_xx."""
    K, fields, _ = permeability(np.zeros(n * n), faces=face_masks(n, inside), return_fields=True)
    w, z, _ = fields[0]
    s = Pe / K[0, 0]
    uu = (w * s).reshape(n, n)
    u = np.concatenate([uu, uu[:, :1]], axis=1)
    v = (z * s).reshape(n, n)
    xc = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xc, xc)
    solid = inside(X, Y)
    return u, v, solid, float(K[0, 0])


def _ucell(u):
    return 0.5 * (u[:, :-1] + u[:, 1:])


# ----------------------------------------------------------------------------- cell problems

def interfacial(n, inside, Pe, k_s=667.0, q=1.0, flow=None, return_field=False):
    """Periodic conjugate problem with uniform heating q in the copper. Returns h_v (bulk and mean references)."""
    u, v, solid, K = flow if flow is not None else cell_flow(n, inside, Pe)
    h = 1.0 / n
    k = np.where(solid, k_s, 1.0)
    A, _, fc = assemble(k, u, v, h, periodic_x=True)
    phi_s = solid.mean()
    gamma = q * phi_s / Pe
    kx = fc["kx"]
    rhs = q * solid - gamma * _ucell(u) + gamma * (np.roll(kx, -1, axis=1) - kx) / h
    theta = _pin_solve(A, rhs.ravel(), 0).reshape(n, n)
    x = (np.arange(n) + 0.5) * h
    T = theta + gamma * x[None, :]
    uc = _ucell(u)
    fl = ~solid
    Ts, Tf = T[solid].mean(), T[fl].mean()
    Tb = (uc[fl] * T[fl]).sum() / uc[fl].sum()
    out = {"Pe": Pe, "phi_s": float(phi_s), "h_v_bulk": float(q * phi_s / (Ts - Tb)),
           "h_v_mean": float(q * phi_s / (Ts - Tf)), "Ts_minus_Tb": float(Ts - Tb),
           "copper_spread": float(T[solid].max() - T[solid].min()),
           "energy_residual": float(abs(q * phi_s - gamma * Pe))}
    if return_field:
        return out, T
    return out


def dispersion(n, inside, Pe, flow=None, return_field=False):
    """Effective conductivity of the water phase along x (copper insulating), including Taylor dispersion."""
    u, v, solid, K = flow if flow is not None else cell_flow(n, inside, Pe)
    h = 1.0 / n
    fl = ~solid
    k = fl.astype(float)
    u = u.copy(); v = v.copy()
    # no advection through faces touching copper (Brinkman leakage is ~1e-8 of the mean)
    left = np.concatenate([np.roll(fl, 1, axis=1), fl[:, -1:]], axis=1)
    right = np.concatenate([fl, fl[:, :1]], axis=1)
    u[~(left & right)] = 0.0
    u[:, -1] = u[:, 0]
    v[~(np.roll(fl, 1, axis=0) & fl)] = 0.0
    A, _, fc = assemble(k, u, v, h, periodic_x=True, active=fl)
    uc = _ucell(u)
    U = uc[fl].mean()                                       # intrinsic mean velocity
    kx = fc["kx"]
    rhs = np.where(fl, -(uc - U) + (np.roll(kx, -1, axis=1) - kx) / h, 0.0)
    first = int(np.flatnonzero(fl.ravel())[0])
    b = _pin_solve(A, rhs.ravel(), first).reshape(n, n)
    b = np.where(fl, b - b[fl].mean(), 0.0)
    dbdx = (b - np.roll(b, 1, axis=1)) / h                  # on the left face of each column
    cond = (kx * (1.0 + dbdx)).sum() / fl.sum()
    disp = -((uc - U) * b)[fl].sum() / fl.sum()
    phi_f = fl.mean()
    out = {"Pe": Pe, "phi_f": float(phi_f), "U_intrinsic": float(U), "K_intrinsic": float(cond + disp),
           "conduction_part": float(cond), "dispersion_part": float(disp),
           "k_f_eff": float(phi_f * (cond + disp))}
    if return_field:
        return out, b
    return out


def spreading(n, inside, Pe, k_s=667.0, flow=None, return_field=False, direction="x"):
    """Heat spreading of the copper-water mixture (copper conducting), flow along x, referenced to the water bulk.
    direction "x" (along the flow):  u.grad b - div(k grad b) = -(u_x - U_D 1_water / phi_f) + div(k e_x),
                                     K_xx = <k (1 + db/dx)> - <(u_x - U_D 1_water / phi_f) b>
    direction "y" (across the flow): u.grad b - div(k grad b) = -u_y + div(k e_y),  K_yy = <k (1 + db/dy)> - <u_y b>
    Superficial values; Pe -> 0 gives the conduction k_eff of Project 2."""
    u, v, solid, K = flow if flow is not None else cell_flow(n, inside, Pe)
    h = 1.0 / n
    k = np.where(solid, k_s, 1.0)
    A, _, fc = assemble(k, u, v, h, periodic_x=True)
    fl = ~solid
    if direction == "x":
        w = _ucell(u) - Pe * fl / fl.mean()
        kf = fc["kx"]
        rhs = -w + (np.roll(kf, -1, axis=1) - kf) / h
    else:
        w = 0.5 * (v + np.roll(v, -1, axis=0))
        kf = fc["ky"]
        rhs = -w + (np.roll(kf, -1, axis=0) - kf) / h
    b = _pin_solve(A, rhs.ravel(), 0).reshape(n, n)
    b -= b.mean()
    ax = 1 if direction == "x" else 0
    dbd = (b - np.roll(b, 1, axis=ax)) / h
    cond = (kf * (1.0 + dbd)).mean()
    disp = -(w * b).mean()
    out = {"Pe": Pe, "K_mix": float(cond + disp), "conduction_part": float(cond), "dispersion_part": float(disp)}
    if return_field:
        return out, b
    return out


# ----------------------------------------------------------------------------- analytical references

def plates_inside(solid_fraction):
    """Copper layers of thickness s (y < s), water gap 1 - s, flow along the layers."""
    return lambda x, y: np.mod(y, 1.0) < solid_fraction


def taylor_aris_plates(gap, U):
    """Intrinsic effective conductivity of plane Poiseuille flow between insulating plates: 1 + Pe_gap^2 / 210."""
    return 1.0 + (U * gap) ** 2 / 210.0


NU_PLATES_BULK = 140.0 / 17.0     # uniform wall heat flux, both walls, bulk temperature, D_h = 2 * gap
NU_PLATES_MEAN = 10.0             # same, referenced to the plain mean water temperature


# ----------------------------------------------------------------------------- device: full resolution vs model

def resolved_strip(n, inside, Pe, q_of_x, n_cells, k_s=667.0, flow=None):
    """Fully resolved strip of n_cells pin cells along x (periodic in y: one row of an infinite plate).
    Inlet water at T = 0, heat q(x) in the copper. Returns per-cell copper mean and water bulk temperature."""
    u1, v1, solid1, K = flow if flow is not None else cell_flow(n, inside, Pe)
    h = 1.0 / n
    nx = n * n_cells
    u = np.concatenate([np.tile(u1[:, :n], (1, n_cells)), u1[:, :1]], axis=1)
    v = np.tile(v1, (1, n_cells))
    solid = np.tile(solid1, (1, n_cells))
    k = np.where(solid, k_s, 1.0)
    A, inlet, _ = assemble(k, u, v, h, periodic_x=False)
    xc = (np.arange(nx) + 0.5) * h
    rhs = (q_of_x(xc)[None, :] * solid).ravel()             # T_in = 0, so the inlet term vanishes
    T = spla.spsolve(A.tocsc(), rhs).reshape(n, nx)
    uc = _ucell(u)
    Ts, Tb, Tmax = [], [], []
    for c in range(n_cells):
        sl = slice(c * n, (c + 1) * n)
        S, W, Uc, Tc = solid[:, sl], ~solid[:, sl], uc[:, sl], T[:, sl]
        Ts.append(Tc[S].mean()); Tmax.append(Tc[S].max())
        Tb.append((Uc[W] * Tc[W]).sum() / Uc[W].sum())
    out_flux = (u[:, nx] * T[:, -1]).sum() * h               # advected out at the outlet (per unit depth)
    out_flux += (2 * k[:, 0] * T[:, 0] / h).sum() * h        # conducted back out through the inlet face
    heat_in = (q_of_x(xc)[None, :] * solid).sum() * h * h
    return {"x": (np.arange(n_cells) + 0.5), "Ts": np.array(Ts), "Tb": np.array(Tb), "Ts_max": np.array(Tmax),
            "T": T, "energy_closure": float((out_flux - heat_in) / heat_in), "unknowns": int(nx * n)}


def homogenised_strip(Pe, phi_s, h_v, k_f_eff, q_of_x, length, m=4000):
    """Two-temperature model on a fine 1-D grid (second-order FV). Returns x, T_b(x), T_s(x).
    Inlet face: T = 0 (conduction over half a cell); outlet face: upwind outflow, no conduction."""
    dx = length / m
    x = (np.arange(m) + 0.5) * dx
    S = phi_s * q_of_x(x)
    a, c = Pe / (2 * dx), k_f_eff / dx**2
    main = np.full(m, 2 * c)
    lo = np.full(m - 1, -a - c)
    up = np.full(m - 1, a - c)
    main[0] = a + 3 * c
    main[-1] = a + c
    A = sp.diags([lo, main, up], [-1, 0, 1], format="csc")
    Tb = spla.spsolve(A, S)
    return x, Tb, Tb + S / h_v
