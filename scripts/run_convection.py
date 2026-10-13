"""Project 2c: convective cell properties of a copper-pin cell and the full-resolution check of the homogenised model.

1. Verification on flow between plates: Nu = 140/17 (bulk) and 10 (mean), Taylor-Aris 1 + Pe^2/210.
2. Pin cell (30 % copper): h_v, water-only spreading k_f,eff and mixture spreading K_mix versus Pe, on 64/128/256 grids.
3. Device check: fully resolved strips with a Gaussian hotspot versus the 1-D two-temperature model built from the
   cell numbers (same 64-pixel cells in both, so only the homogenisation error is measured).
4. Flow check: resolved channel with no-slip walls versus Darcy.
Writes results/convection.json and results/convection_fields.npz (fields for figures).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homog.heat import (NU_PLATES_BULK, NU_PLATES_MEAN, cell_flow, dispersion, homogenised_strip,  # noqa: E402
                        interfacial, plates_inside, resolved_strip, spreading, taylor_aris_plates)
from homog.stokes import channel_flow, cylinder_inside  # noqa: E402

RES = ROOT / "results"
K_S = 667.0                      # copper / water conductivity ratio (400 / 0.6)
SOLID = 0.3
PES = [0.1, 1.0, 3.0, 10.0, 30.0, 100.0]


def scaled(flow, Pe):
    u, v, s, K = flow
    return u * Pe, v * Pe, s, K


t0 = time.time()
out = {"k_s_over_k_f": K_S, "solid_fraction": SOLID}

# 1. plates
ver = []
for n in (32, 64, 128, 256):
    s = 0.5
    ins = plates_inside(s)
    f1 = cell_flow(n, ins, 1.0)
    for Pe in (1.0, 10.0):
        fl = scaled(f1, Pe)
        r = interfacial(n, ins, Pe, k_s=1e6, flow=fl)
        d = dispersion(n, ins, Pe, flow=fl)
        g = 1 - s
        ver.append({"n": n, "Pe": Pe, "Nu_bulk": r["h_v_bulk"] * g, "Nu_mean": r["h_v_mean"] * g,
                    "K_water": d["K_intrinsic"], "K_taylor_aris": taylor_aris_plates(g, d["U_intrinsic"])})
out["plates"] = {"rows": ver, "Nu_bulk_exact": NU_PLATES_BULK, "Nu_mean_exact": NU_PLATES_MEAN}
for Pe in (1.0, 10.0):
    a = [r for r in ver if r["Pe"] == Pe]
    ex = {k: 2 * a[-1][k] - a[-2][k] for k in ("Nu_bulk", "Nu_mean", "K_water")}
    out["plates"][f"richardson_Pe{int(Pe)}"] = ex | {"K_taylor_aris": a[-1]["K_taylor_aris"]}
print("plates done", round(time.time() - t0, 1), flush=True)

# 2. pin cell
ins = cylinder_inside(SOLID)
cell = {}
for n in (64, 128, 256):
    f1 = cell_flow(n, ins, 1.0)
    rows = []
    for Pe in PES:
        fl = scaled(f1, Pe)
        r = interfacial(n, ins, Pe, k_s=K_S, flow=fl)
        d = dispersion(n, ins, Pe, flow=fl)
        m = spreading(n, ins, Pe, k_s=K_S, flow=fl)
        rows.append({"Pe": Pe, "phi_s": r["phi_s"], "h_v": r["h_v_bulk"], "h_v_mean": r["h_v_mean"],
                     "copper_spread": r["copper_spread"], "k_f_eff": d["k_f_eff"], "K_mix": m["K_mix"],
                     "K_mix_conduction": m["conduction_part"], "K_mix_dispersion": m["dispersion_part"]})
    cell[n] = {"K_perm": f1[3], "rows": rows}
    print("cell", n, round(time.time() - t0, 1), flush=True)
out["pin_cell"] = {str(n): c for n, c in cell.items()}
ext = []
for a, b in zip(cell[128]["rows"], cell[256]["rows"]):
    ext.append({k: (2 * b[k] - a[k]) for k in ("h_v", "k_f_eff", "K_mix")} | {"Pe": b["Pe"]})
out["pin_cell"]["richardson"] = ext

# 3. device check (64-pixel cells in both routes)
n = 64
f1 = cell_flow(n, ins, 1.0)
dev = []
fields = {}
for Pe in (1.0, 10.0, 30.0):
    fl = scaled(f1, Pe)
    r = interfacial(n, ins, Pe, k_s=K_S, flow=fl)
    d = dispersion(n, ins, Pe, flow=fl)
    m = spreading(n, ins, Pe, k_s=K_S, flow=fl)
    for w in (1, 2, 4, 8):
        L, x0 = 10 * w, 3 * w

        def q(x, w=w, x0=x0):
            return np.exp(-((x - x0) / w) ** 2)
        t1 = time.time()
        R = resolved_strip(n, ins, Pe, q, L, k_s=K_S, flow=fl)
        tr = time.time() - t1
        row = {"Pe": Pe, "hotspot_cells": w, "n_cells": L, "unknowns_resolved": R["unknowns"],
               "seconds_resolved": round(tr, 2), "energy_closure_resolved": R["energy_closure"],
               "peak_Ts_resolved": float(R["Ts"].max()), "peak_pixel_resolved": float(R["Ts_max"].max())}
        for name, kk in (("water_only", d["k_f_eff"]), ("mixture", m["K_mix"])):
            xh, Tb, Ts = homogenised_strip(Pe, r["phi_s"], r["h_v_bulk"], kk, q, L)
            Tsc, Tbc = np.interp(R["x"], xh, Ts), np.interp(R["x"], xh, Tb)
            row[f"peak_Ts_{name}"] = float(Tsc.max())
            row[f"peak_err_{name}"] = float(Tsc.max() / R["Ts"].max() - 1)
            row[f"max_err_Ts_{name}"] = float(np.abs(Tsc - R["Ts"]).max() / R["Ts"].max())
            row[f"max_err_Tb_{name}"] = float(np.abs(Tbc - R["Tb"]).max() / R["Ts"].max())
            if (Pe, w) in ((1.0, 2), (10.0, 2), (30.0, 2)):
                fields[f"Pe{int(Pe)}_x_{name}"] = xh
                fields[f"Pe{int(Pe)}_Ts_{name}"] = Ts
                fields[f"Pe{int(Pe)}_Tb_{name}"] = Tb
        if w == 2:
            fields[f"Pe{int(Pe)}_cells_x"] = R["x"]
            fields[f"Pe{int(Pe)}_cells_Ts"] = R["Ts"]
            fields[f"Pe{int(Pe)}_cells_Tb"] = R["Tb"]
            fields[f"Pe{int(Pe)}_T"] = R["T"].astype(np.float32)
        dev.append(row)
        print("device", Pe, w, f"{row['peak_err_mixture']:+.3%}", round(time.time() - t0, 1), flush=True)
out["device"] = dev

# 4. channel walls
ch = []
for nr in (1, 2, 4, 8, 16):
    c = channel_flow(n, ins, nr)
    if nr == 4:
        fields["channel_u"] = c["u"].astype(np.float32)
    ch.append({k: c[k] for k in ("n_rows", "Q_resolved", "Q_darcy", "ratio", "mass_max_divergence_rel")})
out["channel"] = ch
out["channel_rows_lost"] = float(np.mean([r["n_rows"] * (1 - r["ratio"]) for r in ch]))
out["runtime_s"] = round(time.time() - t0, 1)
(RES / "convection.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(RES / "convection_fields.npz", **fields)
print(json.dumps(out["pin_cell"]["richardson"], indent=1))
print("rows lost to walls:", out["channel_rows_lost"], "runtime", out["runtime_s"])
