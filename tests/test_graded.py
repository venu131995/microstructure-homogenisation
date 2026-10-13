import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from homog.graded import Plate, Props, build_table, cone_filter, resolved_plate  # noqa: E402

TABLE = build_table(32, np.linspace(0.2, 0.55, 8), G_ref=1800.0)
PROPS = Props(TABLE)


def _plate(nx=6, ny=4, m=2, seed=1):
    rng = np.random.default_rng(seed)
    return Plate(nx, ny, 1 + rng.random(nx * ny), PROPS, Qf=60.0, m=m), 0.25 + 0.25 * rng.random(nx * ny)


def test_adjoint_gradients_match_finite_differences():
    pl, s0 = _plate()
    st, gJ, gG = pl.gradient(s0)
    fJ, fG = [], []
    for k in range(s0.size):
        e = np.zeros(s0.size)
        e[k] = 1e-6
        a, b = pl.solve(s0 + e), pl.solve(s0 - e)
        fJ.append((a["J"] - b["J"]) / 2e-6)
        fG.append((a["G"] - b["G"]) / 2e-6)
    assert np.abs(gJ - fJ).max() < 1e-6 * np.abs(fJ).max()
    assert np.abs(gG - fG).max() < 1e-6 * np.abs(fG).max()


def test_flow_rate_and_energy_balance_of_homogenised_plate():
    pl, s0 = _plate()
    st = pl.solve(s0)
    Fx = st["F"][:pl.N]
    assert abs(Fx[pl.wrap].sum() / pl.Qf - 1) < 1e-12
    # heat out at the outlet + back through the inlet = heat in
    Tw = st["T"]
    cb, ca = pl.xb[pl.wrap], pl.xa[pl.wrap]
    from homog.graded import _B
    Din = 2 * st["kxx"][cb]
    back = (Din * _B(Fx[pl.wrap] / Din)[0] * Tw[cb]).sum()
    out = (Fx[pl.wrap] * Tw[ca]).sum()
    assert abs((out + back) / (pl.Q * pl.h**2).sum() - 1) < 1e-10


def test_uniform_design_pressure_matches_darcy():
    pl, _ = _plate()
    st = pl.solve(np.full(pl.nx * pl.ny, 0.3))
    K = PROPS(np.array([0.3]))[0][0]
    assert abs(st["G"] / (pl.Qf / (K * pl.ny)) - 1) < 1e-10


def test_filter_rows_sum_to_one():
    W = cone_filter(10, 6, 2.0)
    assert np.allclose(np.asarray(W.sum(axis=1)).ravel(), 1.0)


def test_resolved_plate_balances():
    s = np.full(3 * 2, 0.3)
    R = resolved_plate(s, 3, 2, 16, np.ones(6), Qf=4.0)
    assert R["flow"]["mass_max_divergence_rel"] < 1e-10
    assert abs(R["flow"]["power_closure_rel"]) < 1e-8
    assert abs(R["energy_closure"]) < 1e-8
