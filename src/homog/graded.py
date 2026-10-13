"""Project 2d: graded pin-fin cold plate designed with homogenised cell properties, then checked at full resolution.

Plate of nx-by-ny pin cells (cell size 1), one copper pin per cell; the pin area fraction s(x, y) is the design.
Units as in heat.py: water k_f = 1, (rho c)_f = 1, viscosity 1.

Homogenised model (each pin cell split into m-by-m finite volumes, h = 1/m):
  flow  : Darcy, face flux F = K_f (p_a - p_b + G h) on x faces and K_f (p_a - p_b) on y faces; p periodic. The total
          flow rate Qf is fixed (pump), so the mean pressure gradient G is an unknown closed by sum(inlet fluxes) = Qf.
          Pressure drop across the plate = G nx (the constraint). Periodic across the flow (y).
  heat  : water bulk temperature T:  div(F T) - div(K_mix grad T) = Q (chip heat per plate area), with K_xx on x faces
          and K_yy on y faces; exponentially fitted (Scharfetter-Gummel) face fluxes; inlet T = 0, outflow at x = nx.
  copper: T_s = T + Q / h_v(s).
Cell properties come from the 2b/2c cell problems evaluated at the cell Peclet number Pe(s) = K(s) G_ref, i.e. the local
Darcy speed of that pin size under the design pressure gradient (exact at the constraint, G = G_ref).
Objective: p-norm of T_s (smooth stand-in for the peak). Gradients of the objective and of G by the discrete adjoint.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .heat import cell_flow, interfacial, spreading
from .stokes import cylinder_inside


# ----------------------------------------------------------------------------- cell-property table

def cell_properties(n, s, G_ref, k_s=667.0):
    ins = cylinder_inside(s)
    u, v, solid, K = cell_flow(n, ins, 1.0)
    Pe = K * G_ref
    fl = (u * Pe, v * Pe, solid, K)
    r = interfacial(n, ins, Pe, k_s=k_s, flow=fl)
    kx = spreading(n, ins, Pe, k_s=k_s, flow=fl, direction="x")
    ky = spreading(n, ins, Pe, k_s=k_s, flow=fl, direction="y")
    return {"s": float(s), "phi_pixels": r["phi_s"], "K": K, "Pe": float(Pe), "h_v": r["h_v_bulk"],
            "K_xx": kx["K_mix"], "K_yy": ky["K_mix"]}


def build_table(n, s_grid, G_ref, k_s=667.0):
    return [cell_properties(n, s, G_ref, k_s) for s in s_grid]


class Props:
    """Smooth polynomial fits of the cell table in s: log K, h_v, K_xx, K_yy (and their s-derivatives)."""

    def __init__(self, table, deg=5):
        s = np.array([r["s"] for r in table])
        self.c = [np.polyfit(s, np.log([r["K"] for r in table]), deg)] + \
                 [np.polyfit(s, [r[k] for r in table], deg) for k in ("h_v", "K_xx", "K_yy")]

    def __call__(self, s):
        K = np.exp(np.polyval(self.c[0], s))
        out = [K, K * np.polyval(np.polyder(self.c[0]), s)]
        for c in self.c[1:]:
            out += [np.polyval(c, s), np.polyval(np.polyder(c), s)]
        return out                        # K, dK, h_v, dh_v, K_xx, dK_xx, K_yy, dK_yy


class CellwiseProps:
    """Properties given cell by cell (no fit): used to isolate the homogenisation error in the full-resolution check."""

    def __init__(self, R, rows):
        self.v = [R @ np.array([r[k] for r in rows]) for k in ("K", "h_v", "K_xx", "K_yy")]

    def __call__(self, s):
        z = np.zeros_like(s)
        return [self.v[0], z, self.v[1], z, self.v[2], z, self.v[3], z]


def _B(x):
    """Bernoulli function x / (e^x - 1) and its derivative, stable near 0."""
    x = np.asarray(x, float)
    small = np.abs(x) < 1e-4
    xs = np.where(small, 1.0, x)
    em1 = np.expm1(xs)
    B = np.where(small, 1 - x / 2 + x * x / 12, xs / em1)
    dB = np.where(small, -0.5 + x / 6, (em1 - xs * (em1 + 1)) / em1**2)
    return B, dB


def _harm(a, b):
    return 2 * a * b / (a + b), 2 * b * b / (a + b) ** 2, 2 * a * a / (a + b) ** 2


# ----------------------------------------------------------------------------- homogenised plate model

class Plate:
    def __init__(self, nx, ny, Q, props, Qf, m=4, pnorm=10):
        self.nx, self.ny, self.m, self.Qf, self.P, self.props = nx, ny, m, Qf, pnorm, props
        self.NX, self.NY = nx * m, ny * m
        N = self.N = self.NX * self.NY
        self.h = 1.0 / m
        idx = np.arange(N).reshape(self.NY, self.NX)
        J, I = np.divmod(np.arange(N), self.NX)
        self.R = sp.csr_matrix((np.ones(N), (np.arange(N), (J // m) * nx + I // m)), shape=(N, nx * ny))
        self.Q = np.asarray(Q, float).reshape(ny, nx).repeat(m, 0).repeat(m, 1).ravel()
        # x faces: left of each cell (cells in column 0 own the periodic wrap face); y faces: bottom of each cell
        self.xa, self.xb = np.roll(idx, 1, axis=1).ravel(), idx.ravel()
        self.ya, self.yb = np.roll(idx, 1, axis=0).ravel(), idx.ravel()
        self.wrap = np.zeros(N, bool)
        self.wrap[idx[:, 0]] = True
        a = np.concatenate([self.xa, self.ya]); b = np.concatenate([self.xb, self.yb])
        f = np.arange(2 * N)
        self.Dinc = sp.csr_matrix((np.r_[np.ones(2 * N), -np.ones(2 * N)], (np.r_[a, b], np.r_[f, f])), shape=(N, 2 * N))
        self.Grad = sp.csr_matrix((np.r_[np.ones(2 * N), -np.ones(2 * N)], (np.r_[f, f], np.r_[a, b])), shape=(2 * N, N))
        self.j1 = np.r_[np.full(N, self.h), np.zeros(N)]
        self.ein = np.zeros(2 * N)
        self.ein[:N][self.wrap] = 1.0

    # -------------------------------------------------------------- forward
    def solve(self, s_cells):
        N = self.N
        s = self.R @ s_cells
        K, dK, hv, dhv, kxx, dkxx, kyy, dkyy = self.props(s)
        Kx, Kxa, Kxb = _harm(K[self.xa], K[self.xb])
        Ky, Kya, Kyb = _harm(K[self.ya], K[self.yb])
        Kf = np.r_[Kx, Ky]
        App = (self.Dinc @ sp.diags(Kf) @ self.Grad).tolil()
        ApG = self.Dinc @ (Kf * self.j1)
        App[0, :] = 0; App[0, 0] = 1.0; ApG[0] = 0.0          # pressure level: p_0 = 0
        Af = sp.bmat([[App.tocsr(), sp.csr_matrix(ApG[:, None])],
                      [sp.csr_matrix((self.ein * Kf) @ self.Grad), sp.csr_matrix([[self.ein @ (Kf * self.j1)]])]],
                     format="csc")
        rhs = np.zeros(N + 1)
        rhs[N] = self.Qf
        x = spla.spsolve(Af, rhs)
        p, G = x[:N], x[N]
        drive = self.Grad @ p + self.j1 * G
        F = Kf * drive
        Dx, Dxa, Dxb = _harm(kxx[self.xa], kxx[self.xb])
        Dy, Dya, Dyb = _harm(kyy[self.ya], kyy[self.yb])
        A = self._heat_matrix(F, Dx, Dy, kxx)
        T = spla.spsolve(A, self.Q * self.h**2)
        Ts = T + self.Q / hv
        J = np.mean(Ts**self.P) ** (1 / self.P)
        return dict(s=s, K=K, dK=dK, hv=hv, dhv=dhv, kxx=kxx, dkxx=dkxx, kyy=kyy, dkyy=dkyy, Kf=Kf, Kxa=Kxa,
                    Kxb=Kxb, Kya=Kya, Kyb=Kyb, Dx=Dx, Dxa=Dxa, Dxb=Dxb, Dy=Dy, Dya=Dya, Dyb=Dyb, Af=Af, p=p,
                    G=float(G), dp=float(G * self.nx), drive=drive, F=F, A=A, T=T, Ts=Ts, J=float(J))

    def _heat_matrix(self, F, Dx, Dy, kxx):
        N = self.N
        Fx, Fy = F[:N], F[N:]
        inter = ~self.wrap
        rows, cols, vals = [], [], []

        def sg(a, b, Ff, D):                                 # flux a->b = D (B(-P) T_a - B(P) T_b), P = F / D
            Bm, _ = _B(-Ff / D); Bp, _ = _B(Ff / D)
            rows.extend([a, a, b, b]); cols.extend([a, b, a, b]); vals.extend([D * Bm, -D * Bp, -D * Bm, D * Bp])

        sg(self.xa[inter], self.xb[inter], Fx[inter], Dx[inter])
        sg(self.ya, self.yb, Fy, Dy)
        cb, ca = self.xb[self.wrap], self.xa[self.wrap]
        Din = 2 * kxx[cb]                                    # inlet T = 0 at half a cell
        Bp, _ = _B(Fx[self.wrap] / Din)
        rows += [cb, ca]; cols += [cb, ca]; vals += [Din * Bp, np.maximum(Fx[self.wrap], 0.0)]   # inlet, outflow
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(N, N)).tocsc()

    # -------------------------------------------------------------- adjoint gradients of J and of G
    def _to_cells(self, w, ia, ib, wa, wb):
        g = np.zeros(self.N)
        np.add.at(g, ia, w * wa); np.add.at(g, ib, w * wb)
        return g

    def _flow_weights(self, mu):
        mp = mu[:self.N].copy()
        mp[0] = 0.0                                          # the pinned row does not depend on the design
        return self.Dinc.T @ mp + mu[self.N] * self.ein

    def _dK(self, wf, st):
        N = self.N
        return (self._to_cells(wf[:N], self.xa, self.xb, st["Kxa"], st["Kxb"])
                + self._to_cells(wf[N:], self.ya, self.yb, st["Kya"], st["Kyb"])) * st["dK"]

    def gradient(self, s_cells):
        st = self.solve(s_cells)
        N, P = self.N, self.P
        Ts, T, F = st["Ts"], st["T"], st["F"]
        dJdTs = st["J"] ** (1 - P) * Ts ** (P - 1) / N
        lam = spla.spsolve(st["A"].T.tocsc(), dJdTs)
        gJ = -dJdTs * self.Q / st["hv"] ** 2 * st["dhv"]
        Fx, Fy = F[:N], F[N:]
        inter = ~self.wrap
        dRdF = np.zeros(2 * N)

        def face_terms(a, b, Ff, D):
            Pf = Ff / D
            Bm, dBm = _B(-Pf); Bp, dBp = _B(Pf)
            w = lam[a] - lam[b]
            return (w * (-dBm * T[a] - dBp * T[b]),
                    w * (Bm * T[a] - Bp * T[b] + Pf * (dBm * T[a] + dBp * T[b])))

        fF, fD = face_terms(self.xa[inter], self.xb[inter], Fx[inter], st["Dx"][inter])
        dRdF[:N][inter] = fF
        gx = self._to_cells(fD, self.xa[inter], self.xb[inter], st["Dxa"][inter], st["Dxb"][inter])
        fF, fD = face_terms(self.ya, self.yb, Fy, st["Dy"])
        dRdF[N:] = fF
        gy = self._to_cells(fD, self.ya, self.yb, st["Dya"], st["Dyb"])
        cb, ca = self.xb[self.wrap], self.xa[self.wrap]
        Din = 2 * st["kxx"][cb]
        Pin = Fx[self.wrap] / Din
        Bp, dBp = _B(Pin)
        dRdF[:N][self.wrap] += lam[cb] * dBp * T[cb] + lam[ca] * (Fx[self.wrap] > 0) * T[ca]
        np.add.at(gx, cb, 2 * lam[cb] * (Bp - Pin * dBp) * T[cb])
        v = st["Kf"] * dRdF
        mu = spla.spsolve(st["Af"].T.tocsc(), -np.r_[self.Grad.T @ v, self.j1 @ v])
        gJ = gJ - self._dK((dRdF + self._flow_weights(mu)) * st["drive"], st) - gx * st["dkxx"] - gy * st["dkyy"]
        eG = np.zeros(N + 1)
        eG[N] = 1.0
        nu = spla.spsolve(st["Af"].T.tocsc(), eG)
        gG = -self._dK(self._flow_weights(nu) * st["drive"], st)
        return st, self.R.T @ gJ, self.R.T @ gG


# ----------------------------------------------------------------------------- design filter

def cone_filter(nx, ny, radius):
    """Normalised cone filter on the design cells (periodic in y, clipped in x)."""
    J, I = np.divmod(np.arange(nx * ny), nx)
    rows, cols, vals = [], [], []
    r = int(np.ceil(radius))
    for dj in range(-r, r + 1):
        for di in range(-r, r + 1):
            w = radius - np.hypot(di, dj)
            if w <= 0:
                continue
            i2, j2 = I + di, (J + dj) % ny
            ok = (i2 >= 0) & (i2 < nx)
            rows.append(np.arange(nx * ny)[ok]); cols.append((j2 * nx + i2)[ok]); vals.append(np.full(ok.sum(), w))
    W = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny))
    return sp.diags(1 / np.asarray(W.sum(axis=1)).ravel()) @ W


# ----------------------------------------------------------------------------- full resolution of a graded plate

def plate_inside(s_cells, nx, ny):
    r2 = (np.asarray(s_cells, float) / np.pi).reshape(ny, nx)

    def inside(x, y):
        i = np.floor(x).astype(int) % nx
        j = np.floor(y).astype(int) % ny
        dx, dy = np.mod(x, 1.0) - 0.5, np.mod(y, 1.0) - 0.5
        return dx * dx + dy * dy < r2[j, i]
    return inside


def resolved_plate(s_cells, nx, ny, n, Q_cells, Qf, k_s=667.0, alpha_max=1e8):
    """Every pin resolved: periodic Stokes driven by a unit body force, rescaled to the flow rate Qf (Stokes is linear),
    then conjugate heat with inlet T = 0 at x = 0 and outflow at x = nx. Per-cell copper / water-bulk temperatures."""
    from .heat import _ucell, assemble
    from .stokes import _shift2
    NX, NY = nx * n, ny * n
    M = NX * NY
    h = 1.0 / n
    inside = plate_inside(s_cells, nx, ny)
    j, i = np.divmod(np.arange(M), NX)
    s_u = inside(i * h, (j + 0.5) * h).astype(float)
    s_v = inside((i + 0.5) * h, j * h).astype(float)
    I = sp.identity(M, format="csr")
    E, W, Nn, S = _shift2(NX, NY, 1, 0), _shift2(NX, NY, -1, 0), _shift2(NX, NY, 0, 1), _shift2(NX, NY, 0, -1)
    lap = (E + W + Nn + S - 4 * I) / h**2
    Au, Av = -lap + sp.diags(alpha_max * s_u), -lap + sp.diags(alpha_max * s_v)
    Dx, Dy = (E - I) / h, (Nn - I) / h
    Z = sp.csr_matrix((M, M))
    A = sp.bmat([[Au, Z, -Dx.T], [Z, Av, -Dy.T], [Dx, Dy, None]], format="lil")
    A[2 * M, :] = 0
    A[2 * M, 2 * M] = 1.0
    rhs = np.zeros(3 * M)
    rhs[:M] = 1.0
    x = spla.spsolve(A.tocsc(), rhs)
    q1 = x[:M].reshape(NY, NX)[:, 0].sum() * h              # flow rate per unit body force
    G = Qf / q1
    x *= G
    u, v = x[:M].reshape(NY, NX), x[M:2 * M].reshape(NY, NX)
    div = Dx @ x[:M] + Dy @ x[M:2 * M]
    power_in = G * x[:M].mean() * nx * ny
    dissip = (x[:M] @ (Au @ x[:M]) + x[M:2 * M] @ (Av @ x[M:2 * M])) * h * h
    flow = {"G": float(G), "dp": float(G * nx), "mass_max_divergence_rel": float(np.abs(div).max() * h / np.abs(u).max()),
            "power_closure_rel": float((dissip - power_in) / power_in)}
    xc, yc = (np.arange(NX) + 0.5) * h, (np.arange(NY) + 0.5) * h
    X, Y = np.meshgrid(xc, yc)
    solid = inside(X, Y)
    cid = np.floor(Y).astype(int) * nx + np.floor(X).astype(int)
    phi = np.bincount(cid[solid], minlength=nx * ny) / (n * n)
    q = np.where(solid, (np.asarray(Q_cells, float).ravel() / phi)[cid], 0.0)
    k = np.where(solid, k_s, 1.0)
    uu = np.concatenate([u, u[:, :1]], axis=1)
    Ah, _, _ = assemble(k, uu, v, h, periodic_x=False)
    T = spla.spsolve(Ah.tocsc(), q.ravel()).reshape(NY, NX)
    uc = _ucell(uu)
    cnt = np.bincount(cid[solid], minlength=nx * ny)
    Ts = np.bincount(cid[solid], weights=T[solid], minlength=nx * ny) / cnt
    wat = ~solid
    Tb = (np.bincount(cid[wat], weights=(uc * T)[wat], minlength=nx * ny)
          / np.bincount(cid[wat], weights=uc[wat], minlength=nx * ny))
    heat_in = q.sum() * h * h
    out = (uu[:, NX] * T[:, -1]).sum() * h + (2 * k[:, 0] * T[:, 0] / h).sum() * h
    pix_max = np.zeros(nx * ny)
    np.maximum.at(pix_max, cid[solid], T[solid])
    return {"Ts": Ts, "Tb": Tb, "Ts_pixel_max": pix_max, "T": T.astype(np.float32), "solid": solid,
            "u": u.astype(np.float32), "v": v.astype(np.float32), "flow": flow,
            "energy_closure": float((out - heat_in) / heat_in), "unknowns_flow": int(3 * M), "unknowns_heat": int(M)}
