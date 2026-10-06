import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from homog import (Mesh, checkerboard, circle_cell, hashin_shtrikman_2d, homogenise, laminate,  # noqa: E402
                   maximise_conductivity, periodic_filter, symmetrise, voigt_reuss)


def eff(ind, k_hi, k_lo, n):
    return homogenise(k_lo + (k_hi - k_lo) * ind, Mesh.periodic(n))[0]


def test_homogeneous_material_returns_itself():
    K = eff(np.zeros(16 * 16), 1.0, 3.7, 16)
    assert np.allclose(K, 3.7 * np.eye(2))


def test_laminate_is_exact_voigt_and_reuss():
    lam = laminate(30, 0.4)
    K = eff(lam, 1.0, 0.01, 30)
    v, r = voigt_reuss(1.0, 0.01, lam.mean())
    assert K[0, 0] == pytest.approx(v, rel=1e-10)
    assert K[1, 1] == pytest.approx(r, rel=1e-10)
    assert abs(K[0, 1]) < 1e-12


def test_tensor_symmetric_positive_definite_and_within_bounds():
    rng = np.random.default_rng(0)
    ind = (rng.random(24 * 24) < 0.45).astype(float)
    K = eff(ind, 1.0, 0.05, 24)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(K) > 0)
    v, r = voigt_reuss(1.0, 0.05, ind.mean())
    assert r - 1e-12 <= np.linalg.eigvalsh(K).min() and np.linalg.eigvalsh(K).max() <= v + 1e-12


def test_checkerboard_converges_towards_dykhne():
    errs = [eff(checkerboard(n), 10.0, 1.0, n)[0, 0] - np.sqrt(10.0) for n in (16, 32, 64)]
    assert errs[0] > errs[1] > errs[2] > 0


def test_isotropic_inclusions_respect_hashin_shtrikman():
    ind = circle_cell(64, 0.4)
    K = eff(ind, 1.0, 0.01, 64)
    lo, hi = hashin_shtrikman_2d(1.0, 0.01, ind.mean())
    assert K[0, 0] == pytest.approx(K[1, 1], rel=1e-8)
    assert lo <= K[0, 0] <= hi


def test_sensitivity_matches_finite_difference():
    n = 12
    rng = np.random.default_rng(1)
    k = rng.uniform(0.1, 1.0, n * n)
    mesh = Mesh.periodic(n)
    K, E = homogenise(k, mesh)
    grad = (E[:, 0, 0] + E[:, 1, 1]) / (n * n)          # d trace / d k_e
    h = 1e-6
    for e in [0, 17, 80, 143]:
        kp, km = k.copy(), k.copy()
        kp[e] += h; km[e] -= h
        fd = (np.trace(homogenise(kp, mesh)[0]) - np.trace(homogenise(km, mesh)[0])) / (2 * h)
        assert grad[e] == pytest.approx(fd, rel=1e-6)


def test_symmetrise_is_a_projection_and_filter_is_periodic():
    rng = np.random.default_rng(2)
    v = rng.random(10 * 10)
    s = symmetrise(v, 10)
    assert np.allclose(symmetrise(s, 10), s)
    a = s.reshape(10, 10)
    assert np.allclose(a, a.T) and np.allclose(a, a[::-1])
    H = periodic_filter(10, 2.0)
    assert np.allclose(np.asarray(H.sum(axis=1)).ravel(), 1)
    assert H.nnz == 10 * 10 * H[0].nnz                      # identical stencil everywhere (wrap-around)


def test_design_beats_circle_and_approaches_bound():
    d = maximise_conductivity(30, 0.5, iters=150, betas=(1, 2, 4, 8), every=40)
    circle = eff(1 - circle_cell(30, 0.5), 1.0, 1e-3, 30)       # conductor matrix around an insulating disc
    assert np.trace(d.K) > np.trace(circle)
    assert d.bound_fraction > 0.9


def test_local_and_global_flux_and_energy_balance():
    from homog import conservation
    rng = np.random.default_rng(5)
    c = conservation(0.01 + rng.random(24 * 24), Mesh.periodic(24))
    for case in ("x", "y"):
        assert c[case]["nodal_residual_max_rel"] < 1e-10           # local equilibrium at every node
        assert c[case]["hill_mandel_rel"] < 1e-10                  # energy = average flux response
        assert c[case]["section_flux_spread_rel"] < 1e-10          # same heat flux through every section
        assert c[case]["mean_fluctuation_gradient"] < 1e-12        # periodic fluctuation
    assert c["symmetry_rel"] < 1e-12
