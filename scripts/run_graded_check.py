"""Project 2d full-resolution check: draw every pin of a design, solve Stokes + conjugate heat, and compare with the
homogenised model.  usage: python run_graded_check.py <uniform|opt> <pixels per cell>

Two homogenised predictions are compared with the resolved plate:
  design model : the polynomial cell table (128-pixel cells) used by the optimiser;
  same pixels  : cell problems re-solved for every pin on the resolved run's own pixels and pressure gradient,
                 so the difference to the resolved plate is the homogenisation error alone.
Writes results/graded_check/<design>_n<n>.json and .npz.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homog.graded import CellwiseProps, Plate, Props, cell_properties, resolved_plate  # noqa: E402

name, n = sys.argv[1], int(sys.argv[2])
OUT = ROOT / "results" / "graded_check"
OUT.mkdir(parents=True, exist_ok=True)
G = json.loads((ROOT / "results" / "graded.json").read_text())
D = np.load(ROOT / "results" / "graded_designs.npz")
NX, NY, QF = G["nx"], G["ny"], G["Qf"]
s = D["s_opt"] if name == "opt" else D["s_uniform"]
Q = D["Q"]

t0 = time.time()
R = resolved_plate(s, NX, NY, n, Q, QF)
t_res = time.time() - t0
print(name, n, "resolved", round(t_res, 1), "s", R["flow"], R["energy_closure"], flush=True)

design = Plate(NX, NY, Q, Props(G["table"]), Qf=QF, m=G["m"]).solve(s)
uniq, inv = np.unique(np.round(s, 12), return_inverse=True)
rows_u = [cell_properties(n, su, R["flow"]["G"]) for su in uniq]
rows = [rows_u[k] for k in inv]
plate = Plate(NX, NY, Q, None, Qf=QF, m=G["m"])
plate.props = CellwiseProps(plate.R, rows)
same = plate.solve(s)


def per_cell(st, key):
    return np.asarray(plate.R.T @ st[key]) / (G["m"] ** 2)


out = {"design": name, "n": n, "unknowns_flow": R["unknowns_flow"], "unknowns_heat": R["unknowns_heat"],
       "seconds_resolved": round(t_res, 1), "flow": R["flow"], "energy_closure": R["energy_closure"],
       "resolved": {"Ts_max": float(R["Ts"].max()), "Ts_pixel_max": float(R["Ts_pixel_max"].max()),
                    "dp": R["flow"]["dp"]}}
for tag, st in (("design_model", design), ("same_pixels", same)):
    Tsc = per_cell(st, "Ts")
    out[tag] = {"Ts_max": float(Tsc.max()), "dp": st["dp"],
                "peak_err": float(Tsc.max() / R["Ts"].max() - 1),
                "max_cell_err": float(np.abs(Tsc - R["Ts"]).max() / R["Ts"].max()),
                "dp_err": float(st["dp"] / R["flow"]["dp"] - 1)}
(OUT / f"{name}_n{n}.json").write_text(json.dumps(out, indent=1))
np.savez_compressed(OUT / f"{name}_n{n}.npz", Ts=R["Ts"], Tb=R["Tb"], T=R["T"][::max(1, n // 32), ::max(1, n // 32)],
                    solid=R["solid"][::max(1, n // 32), ::max(1, n // 32)],
                    Ts_design=per_cell(design, "Ts"), Ts_same=per_cell(same, "Ts"))
print(json.dumps(out, indent=1))
