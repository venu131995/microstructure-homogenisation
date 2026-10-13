import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from homog import Mesh, homogenise  # noqa: E402
from homog.heat import (NU_PLATES_BULK, NU_PLATES_MEAN, cell_flow, dispersion, homogenised_strip,  # noqa: E402
                        interfacial, plates_inside, resolved_strip, spreading, taylor_aris_plates)
from homog.stokes import channel_flow, cylinder_inside  # noqa: E402


def _plates(n, Pe):
    ins = plates_inside(0.5)
    fl = cell_flow(n, ins, Pe)
    return interfacial(n, ins, Pe, k_s=1e6, flow=fl), dispersion(n, ins, Pe, flow=fl)


def test_plates_nusselt_converges_to_exact():
    (a, _), (b, _) = _plates(64, 1.0), _plates(128, 1.0)
    nu_b = 0.5 * (2 * b["h_v_bulk"] - a["h_v_bulk"])
    nu_m = 0.5 * (2 * b["h_v_mean"] - a["h_v_mean"])
    assert abs(nu_b / NU_PLATES_BULK - 1) < 0.005
    assert abs(nu_m / NU_PLATES_MEAN - 1) < 0.005


def test_plates_taylor_aris():
    (_, a), (_, b) = _plates(64, 10.0), _plates(128, 10.0)
    ex = 2 * b["K_intrinsic"] - a["K_intrinsic"]
    assert abs(ex / taylor_aris_plates(0.5, b["U_intrinsic"]) - 1) < 0.01


def test_interfacial_energy_balance_and_isothermal_pins():
    ins = cylinder_inside(0.3)
    r = interfacial(32, ins, 10.0)
    assert r["energy_residual"] < 1e-12
    assert r["copper_spread"] < 0.05 * r["Ts_minus_Tb"]


def test_mixture_spreading_matches_fem_conduction_at_zero_flow():
    n = 64
    ins = cylinder_inside(0.3)
    m = spreading(n, ins, 1e-6)
    xc = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(xc, xc)
    k = np.where(ins(X, Y), 667.0, 1.0).ravel()
    k_fem = homogenise(k, Mesh.periodic(n))[0][0, 0]
    assert abs(m["conduction_part"] / k_fem - 1) < 0.02
    assert abs(m["dispersion_part"]) < 1e-6


def test_full_resolution_matches_homogenised_model():
    n, Pe, w = 32, 10.0, 4
    ins = cylinder_inside(0.3)
    fl = cell_flow(n, ins, Pe)
    r, m = interfacial(n, ins, Pe, flow=fl), spreading(n, ins, Pe, flow=fl)

    def q(x):
        return np.exp(-((x - 3 * w) / w) ** 2)
    R = resolved_strip(n, ins, Pe, q, 10 * w, flow=fl)
    assert abs(R["energy_closure"]) < 1e-8
    xh, Tb, Ts = homogenised_strip(Pe, r["phi_s"], r["h_v_bulk"], m["K_mix"], q, 10 * w)
    assert abs(np.interp(R["x"], xh, Ts).max() / R["Ts"].max() - 1) < 0.02


def test_channel_walls_cost_a_constant_strip_of_flow():
    ins = cylinder_inside(0.3)
    a, b = channel_flow(32, ins, 2), channel_flow(32, ins, 4)
    assert a["ratio"] < b["ratio"] < 1
    assert abs(2 * (1 - a["ratio"]) - 4 * (1 - b["ratio"])) < 0.02
    assert b["mass_max_divergence_rel"] < 1e-10
