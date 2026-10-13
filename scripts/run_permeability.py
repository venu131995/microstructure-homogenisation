"""Heat vs flow trade-off of cold-plate microstructures: thermal conductivity k_eff (FEM, copper 1 / water 0.01)
and permeability K (periodic Stokes, cell-size units) for the same cells, flow and heat along x."""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from homog import Mesh, circle_cell, homogenise, maximise_conductivity  # noqa: E402
from homog.stokes import balances, cylinder_inside, face_masks, permeability  # noqa: E402

RES = ROOT / "results"
N = 128
K_CU, K_W = 1.0, 0.01


def k_heat(copper):
    return float(homogenise(K_W + (K_CU - K_W) * copper, Mesh.periodic(N))[0][0, 0])


t0 = time.time()
out = {"N": N, "families": {}}
fr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# A. copper pins in water (flow can pass between pins)
rows = []
for s in fr:
    pins = circle_cell(N, s)
    K = permeability(np.zeros(N * N), faces=face_masks(N, cylinder_inside(s)))
    rows.append({"copper": s, "k": k_heat(pins), "K": float(K[0, 0])})
out["families"]["copper pins"] = rows
# B. water holes in copper (isolated holes: no through-flow)
rows = []
for s in fr:
    holes = circle_cell(N, 1 - s)
    K = permeability(1 - holes)
    rows.append({"copper": s, "k": k_heat(1 - holes), "K": float(K[0, 0])})
out["families"]["water holes"] = rows
# C. straight fins along the flow (layers parallel to x)
rows = []
for s in fr:
    j = np.arange(N * N) // N
    fins = ((j + 0.5) / N < s).astype(float)
    K = permeability(fins, faces=face_masks(N, lambda x, y, s=s: y < s))
    rows.append({"copper": s, "k": k_heat(fins), "K": float(K[0, 0]), "K_exact": (1 - s) ** 3 / 12})
out["families"]["straight fins"] = rows
# D. optimised thermal lattices (Project 2 inverse design)
rows = []
for s in (0.3, 0.5, 0.7):
    d = maximise_conductivity(60, s)
    lat = d.rho.reshape(60, 60).repeat(2, 0).repeat(2, 1).ravel()          # 120 x 120 pixels
    K = permeability(lat)                                                   # metal = solid
    k_lat = float(homogenise(K_W + (K_CU - K_W) * lat, Mesh.periodic(120))[0][0, 0])
    rows.append({"copper": float(lat.mean()), "k": k_lat, "K": float(K[0, 0])})
out["families"]["optimised lattice"] = rows
out["balances_pins_30"] = balances(np.zeros(N * N), faces=face_masks(N, cylinder_inside(0.3)))
out["runtime_s"] = round(time.time() - t0, 1)
(RES / "permeability_tradeoff.json").write_text(json.dumps(out, indent=1))
for fam, rs in out["families"].items():
    print(fam, [(r["copper"], round(r["k"], 3) if r["k"] is not None else None, f"{r['K']:.2e}") for r in rs])
