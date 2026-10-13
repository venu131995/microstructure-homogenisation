"""Figure for section 2d: graded pin-fin plate designed with cell properties, checked pin by pin.

Reads results/graded_designs.npz and results/graded_check/{uniform,opt}_n64.npz (no solves).
Writes figures/fig4_graded_plate.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NX, NY = 16, 8

des = np.load(ROOT / "results/graded_designs.npz")
uni = np.load(ROOT / "results/graded_check/uniform_n64.npz")
opt = np.load(ROOT / "results/graded_check/opt_n64.npz")
Q = des["Q"].reshape(NY, NX)
ext = [0, NX, 0, NY]

fig, ax = plt.subplots(3, 1, figsize=(7.2, 8.6), constrained_layout=True)

im = ax[0].imshow(des["s_opt"].reshape(NY, NX), origin="lower", extent=ext, cmap="jet", vmin=0.2, vmax=0.55)
ax[0].set_title("Optimised copper fraction per pin cell (red dashed: hotspot); flow left to right")
fig.colorbar(im, ax=ax[0], shrink=0.9, label="copper fraction s")

vmax = max(uni["T"].max(), opt["T"].max())
for a, d, name in [(ax[1], uni, "Uniform 27.5% pins"), (ax[2], opt, "Graded design")]:
    im = a.imshow(d["T"], origin="lower", extent=ext, cmap="jet", vmin=0, vmax=vmax)
    a.contour(np.linspace(0, NX, d["solid"].shape[1]), np.linspace(0, NY, d["solid"].shape[0]),
              d["solid"].astype(float), levels=[0.5], colors="k", linewidths=0.25)
    a.set_title(f"{name}, every pin resolved (64 px per pin): peak pin T = {d['Ts'].max():.3f}")
    fig.colorbar(im, ax=a, shrink=0.9, label="temperature")
for a in ax:
    a.set_xticks([]), a.set_yticks([])
    # hotspot outline (heat >= 4x): white halo under a bold red dash so it shows on the jet maps
    xf, yf = np.meshgrid(np.linspace(0, NX, 400), np.linspace(0, NY, 200))
    Qf = 1 + 7 * np.exp(-((xf - 6) ** 2 + (yf - 4) ** 2) / 6)        # same hotspot as scripts/run_graded.py
    a.contour(xf, yf, Qf, levels=[4.0], colors="w", linewidths=4.5)
    a.contour(xf, yf, Qf, levels=[4.0], colors="#d7191c", linewidths=3.0, linestyles=[(0, (4, 2.5))])

fig.savefig(ROOT / "figures/fig4_graded_plate.png", dpi=150)
print("peak uniform", uni["Ts"].max(), "graded", opt["Ts"].max(), "change", opt["Ts"].max() / uni["Ts"].max() - 1)
