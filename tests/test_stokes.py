import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from homog.stokes import (balances, cylinder_inside, drummond_tahir, face_masks, permeability,  # noqa: E402
                          poiseuille_layers)


def test_layers_converge_to_poiseuille_first_order():
    err = []
    for n in (32, 64, 128):
        j = np.arange(n * n) // n
        K = permeability(((j + 0.5) / n >= 0.5).astype(float), faces=face_masks(n, lambda x, y: y >= 0.5))
        err.append(K[0, 0] - poiseuille_layers(0.5))
        assert K[1, 1] < 1e-6                         # no flow across the layers
    assert all(e > 0 for e in err)
    rates = np.log2(np.array(err[:-1]) / np.array(err[1:]))
    assert np.all(rates > 0.9)
    richardson = 2 * (err[-1]) - err[-2]
    assert abs(richardson) / poiseuille_layers(0.5) < 0.01


def test_dilute_cylinders_match_drummond_tahir():
    K = permeability(np.zeros(128 * 128), faces=face_masks(128, cylinder_inside(0.05)))
    assert abs(K[0, 0] / drummond_tahir(0.05) - 1) < 0.03


def test_mass_power_and_symmetry_balances():
    b = balances(np.zeros(64 * 64), faces=face_masks(64, cylinder_inside(0.3)))
    assert b["mass_max_divergence_rel"] < 1e-9
    assert abs(b["power_closure_rel"]) < 1e-10
    assert b["K_symmetry"] < 1e-10


def test_isolated_pores_do_not_flow():
    n = 64
    j, i = np.divmod(np.arange(n * n), n)
    r = np.hypot((i + 0.5) / n - 0.5, (j + 0.5) / n - 0.5)
    K = permeability((r > 0.3).astype(float))          # a closed water pocket in solid
    assert abs(K[0, 0]) < 1e-6
