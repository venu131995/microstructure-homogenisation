"""Animation: a periodic microstructure being designed for maximum heat conduction, then heat flowing through it.

Part 1 - the unit cell (tiled 3x3) evolves under MMA from a plain circular hole; the curve shows how close the
effective conductivity gets to the Hashin-Shtrikman upper bound (the theoretical best for any geometry).
Part 2 - heat-flux magnitude in the final design under a left-to-right temperature gradient.
Writes figures/microstructure_movie.mp4 and figures/microstructure_movie.gif.
"""
from __future__ import annotations

import sys
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import animation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from homog import Mesh, element_flux, hashin_shtrikman_2d, homogenise, maximise_conductivity  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
FIG = ROOT / "figures"
N, VOL, K_LO = 60, 0.5, 1e-3


def main() -> None:
    d = maximise_conductivity(N, VOL, snapshot_every=2)
    hs = hashin_shtrikman_2d(1.0, K_LO, VOL)[1]
    frac = np.array([h["trace"] for h in d.history]) / (2 * hs)
    snaps = d.snapshots

    k_final = K_LO + (1 - K_LO) * d.rho
    _, _, fields = homogenise(k_final, Mesh.periodic(N), return_fields=True)
    q = element_flux(k_final, fields, 0)
    qmag = np.hypot(q[:, 0], q[:, 1]).reshape(N, N)

    fig = plt.figure(figsize=(10, 4.6), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25)
    ax_m, ax_c = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    im = ax_m.imshow(np.tile(snaps[0][1].reshape(N, N), (3, 3)), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax_m.set_xticks([]); ax_m.set_yticks([])
    box = plt.Rectangle((N - 0.5, N - 0.5), N, N, fill=False, ec="tab:orange", lw=1.5)
    ax_m.add_patch(box)
    ax_c.axhline(1.0, color="tab:red", ls="--", lw=1, label="Hashin-Shtrikman bound (best possible)")
    line, = ax_c.plot([], [], color="tab:blue", lw=1.8, label="Designed microstructure")
    ax_c.set_xlim(0, len(frac)); ax_c.set_ylim(0.4, 1.05)
    ax_c.set_xlabel("Optimisation iteration"); ax_c.set_ylabel("Effective conductivity / theoretical maximum")
    ax_c.legend(loc="lower right", fontsize=8)
    title = fig.suptitle("", fontsize=11)
    sub = ax_m.set_title("", fontsize=9)

    n_design, n_flux = len(snaps), 45

    def update(f):
        if f < n_design:
            it, rho = snaps[f]
            im.set_data(np.tile(rho.reshape(N, N), (3, 3)))
            im.set_cmap("gray_r"); im.set_clim(0, 1)
            line.set_data(np.arange(it + 1), frac[: it + 1])
            title.set_text(f"Designing a microstructure for maximum heat conduction   iteration {it}")
            sub.set_text("black = metal, white = insulator; orange box = one unit cell")
        else:
            a = min(1.0, (f - n_design) / 15)
            blend = (1 - a) * np.tile(d.rho.reshape(N, N), (3, 3)) + a * np.tile(qmag / qmag.max(), (3, 3))
            im.set_data(blend); im.set_cmap("inferno" if a > 0.5 else "gray_r"); im.set_clim(0, 1)
            line.set_data(np.arange(len(frac)), frac)
            title.set_text(f"Final design: {100 * d.bound_fraction:.1f}% of the theoretical maximum   heat flowing left to right")
            sub.set_text("bright = high heat flux through the metal network")
        return im, line, title, sub

    frames = list(range(n_design + n_flux))
    anim = animation.FuncAnimation(fig, update, frames=frames, interval=60)
    anim.save(FIG / "microstructure_movie.mp4", writer=animation.FFMpegWriter(fps=18, bitrate=2400), dpi=120)
    anim2 = animation.FuncAnimation(fig, update, frames=frames[:n_design:2] + frames[n_design::2], interval=110)
    anim2.save(FIG / "microstructure_movie.gif", writer=animation.PillowWriter(fps=9), dpi=70)
    print("frames", len(frames), "bound fraction", d.bound_fraction)


if __name__ == "__main__":
    main()
