"""Homogenisation study: verification against exact results, optimal unit cells vs Hashin-Shtrikman,
and a physically based material law compared with the RAMP interpolation used in Project 1."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homog import (Mesh, checkerboard, circle_cell, hashin_shtrikman_2d, homogenise, laminate,  # noqa: E402
                   maximise_conductivity, maxwell_garnett_2d, voigt_reuss)

FIG, RES = ROOT / "figures", ROOT / "results"
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
R90 = np.array([[0, -1], [1, 0]])


def ellipse(n, a, b):
    sub = 4
    s = (np.arange(n * sub) + 0.5) / (n * sub) - 0.5
    X, Y = np.meshgrid(s, s)
    return ((X / a) ** 2 + (Y / b) ** 2 <= 1).astype(float).reshape(n, sub, n, sub).mean(axis=(1, 3)).ravel()


def eff(ind, k_hi, k_lo, n):
    return homogenise(k_lo + (k_hi - k_lo) * ind, Mesh.periodic(n))[0]


def main() -> None:
    FIG.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    t0 = time.time()
    out = {}

    # ------------------------------------------------------------------ 1. verification
    lam = laminate(40, 0.4)
    K = eff(lam, 1.0, 0.01, 40)
    v, r = voigt_reuss(1.0, 0.01, lam.mean())
    out["laminate"] = {"k_parallel": K[0, 0], "voigt": v, "k_series": K[1, 1], "reuss": r,
                       "max_rel_error": float(max(abs(K[0, 0] / v - 1), abs(K[1, 1] / r - 1)))}

    ns = [16, 32, 64, 128, 256]
    cb = [float(eff(checkerboard(n), 10.0, 1.0, n)[0, 0]) for n in ns]
    err = np.array(cb) - np.sqrt(10.0)
    out["checkerboard_dykhne"] = {"n": ns, "k_eff": cb, "exact": float(np.sqrt(10.0)), "error": err.tolist(),
                                  "observed_rate": [float(np.log2(err[i] / err[i + 1])) for i in range(len(ns) - 1)]}

    kel = []
    for n in [64, 128, 256]:
        g = ellipse(n, 0.35, 0.18)
        Ka, Kb = eff(g, 10.0, 1.0, n), eff(1 - g, 10.0, 1.0, n)        # phases swapped
        rhs = 10.0 * R90 @ np.linalg.inv(Kb) @ R90.T
        kel.append({"n": n, "K_xx": Ka[0, 0], "K_yy": Ka[1, 1], "max_rel_error": float(np.abs(Ka - rhs).max() / np.abs(Ka).max())})
    out["keller_reciprocal_theorem"] = kel

    circ = []
    for f in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        ind = circle_cell(128, f)
        k = float(eff(ind, 1.0, 0.01, 128)[0, 0])
        lo, hi = hashin_shtrikman_2d(1.0, 0.01, ind.mean())
        circ.append({"f": float(ind.mean()), "k_eff": k, "maxwell_garnett": maxwell_garnett_2d(1.0, 0.01, ind.mean()),
                     "hs_lower": lo, "hs_upper": hi, "within_bounds": bool(lo - 1e-9 <= k <= hi + 1e-9)})
    out["circular_inclusions_conductive_in_insulating"] = circ

    # ------------------------------------------------------------------ 2. optimal unit cells
    designs = {}
    for vol in (0.3, 0.5, 0.7):
        d = maximise_conductivity(60, vol, snapshot_every=0)
        designs[vol] = d
    out["optimal_cells"] = [{"vol": float(d.rho.mean()), "K_xx": float(d.K[0, 0]), "K_yy": float(d.K[1, 1]),
                             "hs_upper": d.hs_upper, "fraction_of_hs_upper_trace": d.bound_fraction,
                             "voigt": voigt_reuss(1.0, 1e-3, d.rho.mean())[0]} for d in designs.values()]

    # ------------------------------------------------------------------ 3. material law vs RAMP (Project 1 link)
    # Project 1 mixes metal (k=1) and coolant (k=0.01). What effective conductivity does a cell with a given
    # coolant fraction actually have? Compare coolant holes in metal, the HS bounds and the RAMP law (q = 8).
    phis = np.linspace(0.05, 0.85, 17)
    law = []
    for phi in phis:
        holes = circle_cell(96, phi)                                         # 1 = coolant
        k_holes = float(eff(1 - holes, 1.0, 0.01, 96)[0, 0])
        lo, hi = hashin_shtrikman_2d(1.0, 0.01, 1 - holes.mean())
        s = 1 - holes.mean()
        ramp = 0.01 + 0.99 * s / (1 + 8 * (1 - s))
        law.append({"coolant_fraction": float(holes.mean()), "k_coolant_holes": k_holes, "hs_upper": hi, "hs_lower": lo,
                    "ramp_q8": ramp})
    out["material_law_vs_ramp"] = law
    out["runtime_s"] = round(time.time() - t0, 1)
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float))

    # ------------------------------------------------------------------ figures
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1))
    h = 1 / np.array(ns)
    axs[0].loglog(h, err, "o-"); axs[0].set_xlabel("element size h"); axs[0].set_ylabel("error vs exact sqrt(k1 k2)")
    axs[0].set_title(f"Checkerboard (Dykhne): rate ~ h^{np.mean(out['checkerboard_dykhne']['observed_rate']):.2f}")
    axs[1].loglog([1 / k["n"] for k in kel], [k["max_rel_error"] for k in kel], "s-", color="tab:green")
    axs[1].set_xlabel("element size h"); axs[1].set_ylabel("relative error"); axs[1].set_title("Keller reciprocal theorem")
    f = [c["f"] for c in circ]
    axs[2].fill_between(f, [c["hs_lower"] for c in circ], [c["hs_upper"] for c in circ], color="0.88", label="HS bounds")
    axs[2].plot(f, [c["maxwell_garnett"] for c in circ], "k--", lw=1, label="Maxwell-Garnett")
    axs[2].plot(f, [c["k_eff"] for c in circ], "o", ms=4, label="FEM, square array of discs")
    axs[2].set_yscale("log"); axs[2].set_xlabel("inclusion fraction"); axs[2].set_ylabel("k_eff"); axs[2].legend(fontsize=7)
    axs[2].set_title("Conductive discs in insulating matrix")
    fig.tight_layout(); fig.savefig(FIG / "fig1_verification.png"); plt.close(fig)

    fig, axs = plt.subplots(2, 3, figsize=(9, 6))
    for col, (vol, d) in enumerate(designs.items()):
        cell = d.rho.reshape(60, 60)
        axs[0, col].imshow(cell, cmap="gray_r", interpolation="nearest"); axs[0, col].set_title(f"unit cell, {int(vol * 100)}% conductor")
        axs[1, col].imshow(np.tile(cell, (3, 3)), cmap="gray_r", interpolation="nearest")
        axs[1, col].set_title(f"{100 * d.bound_fraction:.1f}% of HS upper bound")
        for a in axs[:, col]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Optimised periodic microstructures for maximum isotropic conductivity (contrast 1000:1)")
    fig.tight_layout(); fig.savefig(FIG / "fig2_optimal_cells.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    cf = [r["coolant_fraction"] for r in law]
    ax.fill_between(cf, [r["hs_lower"] for r in law], [r["hs_upper"] for r in law], color="0.9", label="HS bounds (any geometry)")
    ax.plot(cf, [r["hs_upper"] for r in law], color="0.5", lw=1)
    ax.plot(cf, [r["k_coolant_holes"] for r in law], "o-", ms=3, label="FEM: circular coolant holes in metal")
    ax.plot(cf, [r["ramp_q8"] for r in law], "s--", ms=3, color="tab:red", label="RAMP q = 8 (Project 1 interpolation)")
    ax.set_yscale("log"); ax.set_xlabel("coolant volume fraction"); ax.set_ylabel("effective conductivity (metal = 1)")
    ax.legend(fontsize=7); ax.set_title("Homogenised material law vs topology-optimisation interpolation")
    fig.tight_layout(); fig.savefig(FIG / "fig3_material_law_vs_ramp.png"); plt.close(fig)

    print(json.dumps({k: out[k] for k in ["laminate", "checkerboard_dykhne", "keller_reciprocal_theorem", "optimal_cells", "runtime_s"]},
                     indent=1, default=float))
    print([(round(r["coolant_fraction"], 2), round(r["k_coolant_holes"], 3), round(r["hs_upper"], 3), round(r["ramp_q8"], 3)) for r in law])
    print([(round(c["f"], 2), round(c["k_eff"], 4), round(c["maxwell_garnett"], 4), c["within_bounds"]) for c in circ])


if __name__ == "__main__":
    main()
