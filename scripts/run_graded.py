"""Project 2d: graded pin-fin cold plate designed from homogenised cell properties.

Plate of 16 x 8 pin cells, 8x Gaussian hotspot, fixed coolant flow rate, periodic across the flow.
Design: pin fraction s in [0.20, 0.55] per cell (cone filter, radius 2 cells).
Problem: minimise the p-norm (p = 10) of the copper temperature subject to pressure drop <= that of uniform 30 % pins.
Solver: SLSQP with discrete-adjoint gradients of the objective and of the pressure drop.
Writes results/graded.json and results/graded_designs.npz.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homog.graded import Plate, Props, build_table, cone_filter  # noqa: E402
from homog.heat import cell_flow  # noqa: E402
from homog.stokes import cylinder_inside  # noqa: E402

RES = ROOT / "results"
NX, NY, QF, M, N_CELL = 16, 8, 160.0, 4, 128
S_LO, S_HI, S_REF, RADIUS = 0.20, 0.55, 0.30, 2.0
X, Y = np.meshgrid(np.arange(NX) + 0.5, np.arange(NY) + 0.5)
Q = (1 + 7 * np.exp(-((X - 6) ** 2 + (Y - 4) ** 2) / 6)).ravel()

t0 = time.time()
K_ref = cell_flow(N_CELL, cylinder_inside(S_REF), 1.0)[3]
G_MAX = QF / (K_ref * NY)                       # pressure gradient of uniform 30 % pins at this flow rate
table = build_table(N_CELL, np.linspace(S_LO, S_HI, 8), G_ref=G_MAX)
props = Props(table)
plate = Plate(NX, NY, Q, props, Qf=QF, m=M)
print("table", round(time.time() - t0, 1), flush=True)

mcheck = {m: float(Plate(NX, NY, Q, props, Qf=QF, m=m).solve(np.full(NX * NY, S_REF))["Ts"].max()) for m in (2, 4, 8)}

scan = []
for s in np.linspace(S_LO, S_HI, 15):
    st = plate.solve(np.full(NX * NY, s))
    scan.append({"s": float(s), "Ts_max": float(st["Ts"].max()), "J": st["J"], "dp": st["dp"],
                 "feasible": bool(st["G"] <= G_MAX * (1 + 1e-9))})
best = min((r for r in scan if r["feasible"]), key=lambda r: r["Ts_max"])
print("best feasible uniform", best, flush=True)

W = cone_filter(NX, NY, RADIUS)


def design(r):
    return S_LO + (S_HI - S_LO) * (W @ r)


cache, hist = {}, []


def ev(r):
    k = r.tobytes()
    if k not in cache:
        cache.clear()
        cache[k] = plate.gradient(design(r))
        st = cache[k][0]
        hist.append({"J": st["J"], "Ts_max": float(st["Ts"].max()), "dp": st["dp"], "s": design(r).copy()})
    return cache[k]


cons = [{"type": "ineq", "fun": lambda r: 1 - ev(r)[0]["G"] / G_MAX,
         "jac": lambda r: -(S_HI - S_LO) * (W.T @ ev(r)[2]) / G_MAX}]
r0 = np.full(NX * NY, (S_REF - S_LO) / (S_HI - S_LO))
res = minimize(lambda r: ev(r)[0]["J"], r0, jac=lambda r: (S_HI - S_LO) * (W.T @ ev(r)[1]), method="SLSQP",
               bounds=[(0, 1)] * (NX * NY), constraints=cons, options={"maxiter": 500, "ftol": 1e-11})
s_opt = design(res.x)
st_opt = plate.solve(s_opt)
s_uni = np.full(NX * NY, best["s"])
st_uni = plate.solve(s_uni)
out = {"nx": NX, "ny": NY, "Qf": QF, "m": M, "cell_pixels": N_CELL, "s_bounds": [S_LO, S_HI], "filter_radius": RADIUS,
       "G_max": G_MAX, "dp_max": G_MAX * NX, "table": table, "model_Ts_max_vs_m": {str(k): v for k, v in mcheck.items()},
       "uniform_scan": scan, "best_uniform": best,
       "optimised": {"Ts_max": float(st_opt["Ts"].max()), "J": st_opt["J"], "dp": st_opt["dp"],
                     "dp_ratio": st_opt["G"] / G_MAX, "s_min": float(s_opt.min()), "s_max": float(s_opt.max()),
                     "iterations": int(res.nit), "evaluations": len(hist), "message": str(res.message)},
       "gain_peak": float(st_opt["Ts"].max() / st_uni["Ts"].max() - 1), "runtime_s": round(time.time() - t0, 1)}
(RES / "graded.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(RES / "graded_designs.npz", Q=Q, s_uniform=s_uni, s_opt=s_opt,
                    hist_s=np.array([h["s"] for h in hist]), hist_J=np.array([h["J"] for h in hist]),
                    hist_Tmax=np.array([h["Ts_max"] for h in hist]), hist_dp=np.array([h["dp"] for h in hist]),
                    Ts_opt=st_opt["Ts"], Ts_uni=st_uni["Ts"], T_opt=st_opt["T"], T_uni=st_uni["T"],
                    F_opt=st_opt["F"], F_uni=st_uni["F"])
print(json.dumps({k: out[k] for k in ("model_Ts_max_vs_m", "best_uniform", "optimised", "gain_peak", "runtime_s")},
                 indent=1))
