# Computational Homogenisation & Inverse Microstructure Design (FEM)

The microstructures inside a cold plate (pin-fin arrays, lattices, channel networks) are far too fine
to mesh in a full-device simulation. Instead, each region is described by **effective properties**
computed on a single periodic unit cell. This project builds that capability from scratch:

1. **A periodic finite-element homogenisation code** (bilinear Q1 elements) for the effective
   conductivity tensor of any two-phase microstructure.
2. **Verification against exact results:** laminates, Dykhne's checkerboard, Keller's reciprocal
   theorem and the Hashin–Shtrikman bounds.
3. **Inverse homogenisation.** Topology optimisation of the unit cell (adjoint sensitivities, MMA,
   periodic filter, D4 symmetry) to design microstructures that approach the theoretical maximum
   conductivity.
4. **A link to Project 1.** The homogenised material law is compared with the RAMP interpolation used
   in the cold-plate optimiser.

![microstructure movie](figures/microstructure_movie.gif)

*`figures/microstructure_movie.mp4`: a 3 × 3 tiling of the unit cell evolving from a plain circular hole
into the optimised network, with its conductivity approaching the Hashin–Shtrikman bound. The final
frames show the heat flux through the design.*

## Method

For a unit cell of n × n pixels, the effective tensor is

```
K_ij = (1/|Y|) Σ_e (χ⁰_i − χ_i)_eᵀ  k_e K⁰_e  (χ⁰_j − χ_j)_e ,      K χ_i = F_i  (periodic)
```

- **Periodicity:** opposite boundary nodes are identified, so the mesh is a torus. One sparse LU
  factorisation serves both load cases.
- **Sensitivity:** the problem is self-adjoint, so dK_ij/dρ_e is a per-element energy product, with no
  extra solve. It is checked against finite differences to 10⁻⁶.
- **Design:**
  - SIMP interpolation (p = 3), contrast 1000:1 between conductor and insulator;
  - periodic density filter and Heaviside projection (β up to 32);
  - eight-fold square symmetry, which makes the 2-D design isotropic;
  - MMA, maximising K_xx + K_yy at a fixed conductor fraction;
  - starting from a circular hole, since a uniform grey start is a stationary point.

## Results

All numbers come from `results/summary.json`, produced by `scripts/run_homog.py` (≈ 2 min).

### 1. Verification

| Test | Exact result | FEM |
|---|---|---|
| Laminate, along / across layers | Voigt / Reuss means | **exact to 10⁻¹⁰** |
| Checkerboard, contrast 10 (Dykhne) | k = √(k₁k₂) = 3.162 | 3.444 → 3.193 (n = 16 → 256); converges at h^0.79 because of corner singularities |
| Keller reciprocal theorem, ellipse | K(k₁,k₂) = k₁k₂ R K(k₂,k₁)⁻¹ Rᵀ | error 2.3% → 1.1% → 0.58% (first-order convergence) |
| Conductive discs, f = 5–70% | inside the HS bounds; near Maxwell–Garnett when dilute | all inside the bounds; 1–3% above MG for f ≤ 0.4, rising with f as discs interact |

These checks are exact theorems, not "looks reasonable" comparisons, so they test the periodic boundary
conditions, assembly and energy formula independently (`fig1_verification.png`).

### 2. Optimised microstructures vs. the theoretical limit

> Slide version (copper 1 / water 0.01, 30 % copper, 60² cells): the optimised lattice reaches k_eff = 0.184, 99.0 % of the
> HS bound (0.186), against 0.019 for 30 % copper pins (9.6×). Verification at the same contrast is in
> `results/verification_copper_water.json`: stripes exact; pins → Rayleigh series 0.01836 (0.6 % at 384²);
> checkerboard → √(k₁k₂) = 0.1 (0.136 at 512², slow because of the corner singularity).

No two-phase isotropic microstructure can conduct better than the Hashin–Shtrikman upper bound.

| Conductor fraction | Optimised cell | Simple circular hole | Optimised advantage |
|---|---|---|---|
| **30%** | **98.9%** of the bound | 76.5% of the bound | **+29% conductivity** |
| 50% | 98.1% | 97.3% | +0.8% |
| 70% | 99.5% | 99.8% | none (the circle is already near-optimal) |

The optimiser matters most when material is scarce. At 30% conductor, a circular hole leaves thin necks
that choke heat flow. The optimiser instead builds a diagonal lattice of evenly thick ligaments that
reaches 99% of the theoretical maximum. At high conductor fractions a circular hole is already essentially
optimal, and the optimiser rediscovers it (`fig2_optimal_cells.png`).

### 3. What this means for topology optimisation (link to Project 1)

The cold-plate optimiser treats partly-filled cells with a RAMP interpolation of conductivity. Here is
what a real microstructure with the same coolant fraction delivers (`fig3_material_law_vs_ramp.png`):

| Coolant fraction | 10% | 30% | 50% | 70% |
|---|---|---|---|---|
| FEM: circular coolant holes in metal | 0.82 | 0.55 | 0.33 | 0.15 |
| HS upper bound (best possible) | 0.82 | 0.55 | 0.34 | 0.19 |
| RAMP q = 8 (Project 1) | 0.51 | 0.21 | 0.11 | 0.06 |

RAMP underestimates achievable conductivity by **1.6–3×**. That is intended: it is a penalty that pushes
designs to pure 0/1. It is not a material law. A homogenisation-based (multiscale) optimiser could
instead use graded microstructures and their real effective properties. The holes curve also shows the
**percolation collapse** above ~78.5% coolant, where neighbouring holes touch and the metal network
breaks (conductivity falls from 0.15 to 0.02).

## Conservation and confidence

`conservation()` in `fem.py` checks the solved cell problems (`results/conservation.json`, all three optimised cells):

| Check | Meaning | Worst case |
|---|---|---|
| Nodal residual of K χ = F | local equilibrium (discrete ∇·q = 0) at every node | 1 × 10⁻¹⁴ |
| Hill–Mandel consistency | K from the energy integral = volume-averaged flux response | 4 × 10⁻¹⁵ |
| Section flux | identical heat flux through every column (row) of the cell | 5 × 10⁻¹⁵ |
| Periodicity | the fluctuation χ adds no mean gradient | 5 × 10⁻¹⁹ |
| Symmetry | K_xy = K_yx | 7 × 10⁻¹⁸ |

Together with the exact-theorem verification above (laminates exact to 10⁻¹⁰, Dykhne, Keller, HS bounds),
finite-difference-checked sensitivities, and a unit test enforcing these balances, this gives high confidence that
the effective properties are right.

## 2b. Permeability of the same cells (periodic Stokes)

`src/homog/stokes.py` solves the periodic Stokes cell problem on the same pixel grid:
−μ∇²w + α(x)w + ∇π = e_i, ∇·w = 0, with solid pixels blocked by a Brinkman penalty (α = 10⁸). The permeability is
K_ij = ⟨w_i·e_j⟩, the cell-averaged velocity, in units of (cell size)². It uses a staggered MAC grid with a direct
sparse solve. The pressure gradient is the exact negative transpose of the divergence, so the discrete power balance
holds exactly. For smooth geometries, faces are blocked when their centre lies inside the solid. Convergence is first
order (pixel walls), so values are Richardson-extrapolated from 256² and 512² meshes (run in parallel on a compute server,
`results/permeability_convergence/`).

| Check | Reference | Result |
|---|---|---|
| Flow between plates (exact K = φ³/12) | 0.010417 | 0.010417 (0.0 %) |
| Square cylinder array, 5 / 10 / 20 % solid | Drummond & Tahir (1984, dilute) | −0.0 / +0.1 / −1.3 % |
| Square cylinder array, 50 / 60 % solid | Gebart (1992, dense) | −9 / −7 % |
| 30–40 % solid | neither formula is reliable there | solver is 11–14 % below both; not independently verified |
| Mass (divergence), power in = dissipation, K symmetry | — | 10⁻¹³ / 10⁻¹⁵ / 10⁻¹⁴ |
| Sealed water pockets | K = 0 | 10⁻⁷–10⁻⁸ (penalty leakage) |

**Heat vs flow** (`scripts/run_permeability.py` → `results/permeability_tradeoff.json`, 128² cells, flow and heat along
x; copper k = 1, water 0.01):
- **Copper pins in water:** flow passes (K up to 4×10⁻²), but heat conduction stays poor (k_eff 0.01–0.08), because
  the copper is disconnected.
- **Water holes in copper, and the optimised thermal lattices:** heat conducts well (k_eff up to 0.55), but the
  water pockets are sealed (K ≈ 0).
- **Straight fins:** both at once (k_eff ≈ copper fraction, K ≈ (1−s)³/12), but only along the fins.

In 2-D, copper and water cannot both connect in every direction, which is why straight channels and fins are the
default. A cold-plate microstructure has to be designed for both properties together.

## 2c. Convection in the cell, and a full-resolution check

`src/homog/heat.py` adds steady heat transport by the 2b flow on the same pixel grid. It uses cell-centred finite volumes,
MAC-face velocities, harmonic-mean face conductivities and central advection. Units are cell size L = 1, water k_f = 1 and
(ρc)_f = 1, so the Darcy velocity equals the cell Péclet number Pe. The cell is copper pins (30 %) in water, with
k_s/k_f = 667 (copper/water).

Three periodic cell problems give the device-scale numbers:
- **h_v (pin → water)**: uniform heat q in the copper and a periodically developed temperature T = γx + θ, with γ = qφ_s/U_D
  from the energy balance. h_v = qφ_s / (⟨T⟩_copper − T_bulk,water).
- **k_f,eff (water only)**: a Taylor-dispersion closure with the copper insulating.
- **K_mix (copper + water)**: the same closure with the copper conducting, referenced to the water bulk. At Pe → 0 it equals
  the conduction k_eff of Project 2.

Device model (two-temperature, 1-D along the flow; pins are disconnected, so copper does not conduct in-plane):
U_D dT_b/dx = K d²T_b/dx² + φ_s q(x), with T_copper = T_b + φ_s q / h_v.

**Verification** (`scripts/run_convection.py` → `results/convection.json`; first-order pixel walls, so values are
Richardson-extrapolated from 128² and 256²):

| Check | Exact | Result |
|---|---|---|
| Plates, uniform wall flux, Nu (bulk, D_h = 2·gap) | 140/17 = 8.235 (Shah & London) | 8.237 |
| Plates, Nu referred to the mean water temperature | 10 | 10.002 |
| Taylor–Aris between plates, Pe_gap ≈ 7.5 | 1 + Pe²/210 = 1.4762 | 1.4749 (−0.09 %) |
| K_mix at Pe → 0 vs Project 2 conduction FEM (128²) | 1.866 | 1.858 (FV vs FEM, −0.4 %) |
| Energy balance, every resolved strip (incl. heat conducted back out of the inlet) | 0 | ≤ 1.4×10⁻¹⁰ |
| Mass (divergence) in the walled channels | 0 | ≤ 10⁻¹² |

**Pin cell results** (extrapolated):

| Pe | 0.1 | 1 | 3 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|
| h_v | 13.6 | 13.7 | 13.9 | 14.4 | 14.7 | 14.8 |
| K_mix | 1.86 | 1.87 | 1.94 | 2.71 | 9.17 | 80.1 |
| k_f,eff (water only) | 0.54 | 0.57 | 0.77 | 2.20 | 11.5 | 94.3 |

h_v is almost independent of the flow (Stokes regime, like a fully developed laminar Nu). Spreading along the flow grows
roughly as Pe² above Pe ≈ 10 (dispersion).

**Cell route vs full resolution.** A strip of 10w cells (w = hotspot width in cells; Gaussian heating in the pins; inlet
water at T = 0) is solved with every pin resolved (64 pixels per cell, up to 327 680 unknowns, under 1.5 s each). The same
64-pixel cells are used for the cell values, so the comparison measures only the homogenisation error. Worst
copper-temperature error over the strip, with K_mix:

| hotspot width (cells) | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| Pe = 1 | 2.2 % | 0.87 % | 0.27 % | 0.08 % |
| Pe = 10 | 6.9 % | 2.6 % | 1.0 % | 0.46 % |
| Pe = 30 | 9.7 % | 6.6 % | 3.3 % | 1.6 % |

- **The error falls steadily as the hotspot gets wider, i.e. as scale separation improves.** Faster flow needs wider
  hotspots, because heat is carried further before it leaves the pins.
- **At slow flow the copper must be included in the spreading.** At Pe = 1, the water-only k_f,eff misses by up to 27 %,
  because heat also crosses water → pin → water. K_mix fixes this (2.2 %).
- **At Pe ≥ 10 the two choices are within a few per cent of each other.** At Pe = 10 the water-only value is slightly
  closer, by 1–2 %, so neither closure is exact there.

**Flow next to walls.** n pin rows between two no-slip walls, fully resolved, carry Q = K (n − 0.71): the walls cost a
constant 0.71 rows of flow. Darcy with the cell K is therefore 29 % high for one row, and 4 % high for 16 rows. Near walls,
a Brinkman correction or a resolved boundary row is needed.

Limitations: Stokes flow only, so Pe = Re·Pr ≲ 10–70 for water. Real cold plates (Re ~ 10²) need Navier–Stokes cell
solutions. The model is 2-D: heat enters the pins uniformly instead of from a base plate, and the copper is disconnected
in-plane.

## 2d. Graded pin-fin plate: design with cell properties, check with every pin solved

`src/homog/graded.py` closes the multiscale loop. The cell problems of 2b/2c become a lookup table, an optimiser grades
the pin size across a plate, and the optimised plate is drawn pin by pin and solved at full resolution.

**Problem** (`scripts/run_graded.py` → `results/graded.json`):
- Plate of 16 × 8 pin cells, with an 8× Gaussian hotspot on a uniform background.
- Fixed coolant flow rate, periodic across the flow.
- Design: pin copper fraction s ∈ [0.20, 0.55] in every cell, with a cone filter of radius 2 cells (keeps the grading
  slow, which homogenisation needs).
- Minimise the peak copper temperature (p-norm, p = 10), keeping the pressure drop no higher than a plate of uniform
  30 % pins.
- **Cell table:** 8 pin sizes on 128² cells give K, h_v, K_xx (along flow) and K_yy (across flow). Each is evaluated at
  that pin size's own cell Péclet number, Pe = K(s)·G at the pressure-drop limit, from 2 (dense) to 40 (sparse).
  The table is fitted with polynomials in s.
- **Plate model:** Darcy, with the pressure gradient G as an unknown closed by the flow-rate condition. Water bulk
  temperature by finite volumes (exponentially fitted fluxes, anisotropic spreading). Copper temperature
  T_s = T_b + Q/h_v.
- **Gradients:** the discrete adjoint of the bordered flow system and the heat equation, i.e. one flow solve, one heat
  solve and three adjoint solves per iteration. They match central finite differences to 10⁻⁹ (objective and pressure
  drop, `tests/test_graded.py`).
- **Optimiser:** SLSQP (442 iterations, 49 s). Refining the plate model to 2, 4 or 8 sub-cells per pin changes the peak
  by under 0.01 %.

**Result (cell route):**
- The best uniform plate that meets the pressure limit (s = 0.275) has a peak of 2.051. The graded plate has 1.645
  (−19.8 %) at the limit.
- Even uniform 55 % pins, which need 8.5× the pressure drop, only reach 1.867.
- The optimiser keeps sparse pins in the hotspot row and builds dense pin "walls" above and below it. These funnel
  coolant into the hotspot row: flow steering, like the channels of Project 1, but made from pin sizes.

**Full-resolution check** (`scripts/run_graded_check.py`, on a compute server → `results/graded_check/`):
- Every pin is drawn as pixels. Periodic Stokes is solved on 48 or 64 pixels per pin, plus conjugate heat (copper 667×
  water). That is up to 2.1 M unknowns, 30 min per plate on one core.
- "Same pixels" re-solves the cell problem of every pin on the run's own grid and pressure gradient, so its error is
  the homogenisation error alone.

| 64 px per pin | uniform (s = 0.275) | graded |
|---|---|---|
| Peak pin temperature, full resolution | 2.020 | 1.665 (**−17.6 %**) |
| Cell route, same pixels | 2.009 (−0.6 %) | 1.636 (−1.7 %) |
| Cell route used for the design | 2.006 (−0.7 %) | 1.633 (−1.9 %) |
| Worst single-cell error (same pixels) | 2.4 % | 4.9 % |
| Pressure drop, cell route vs resolved | 0.0 % | −0.3 % |
| Peak change 48 → 64 px per pin | 0.0 % | −0.2 % |

- The cell route predicted the gain as −18.6 % (cell averages); the real plates give −17.6 %.
- The design table (128² cells) over-predicts the pressure drop by 7.1 % / 5.9 % at 48 px, and by 1.4 % / 3.6 % at
  64 px. The gap shrinks with refinement, so it is pixel error in the resolved runs, not a modelling error.
- Balances (resolved): mass divergence ≤ 6×10⁻¹¹, pump power = viscous dissipation to 5×10⁻¹⁴, heat in = heat out to
  7×10⁻¹².

**Limitations:**
- Stokes flow, so Re ≪ 1. Real cold plates run at Re ~ 10².
- 2-D: heat enters the pins uniformly instead of conducting up from a base.
- Periodic across the flow (no side walls); 2c measured a 0.7-row wall loss.
- Cell properties are evaluated at the Pe of each pin size at the pressure-drop limit, not at the local velocity.

## Limitations and next steps

- Conductivity and permeability are computed separately. A two-objective (heat + flow) inverse design and a graded,
  multiscale cold-plate model are the natural next steps.
- Permeability is first-order accurate on pixel walls (extrapolation needed). It is unverified in the 30–40 % range.
- 2-D square pixels, which is the source of the slow checkerboard convergence. Body-fitted or adaptive
  meshes would converge faster.
- Single-objective design. A natural extension is a two-objective conductivity–permeability design,
  mapping the trade-off that a cold-plate microstructure must make.

## Files

| File | Content |
|---|---|
| `src/homog/fem.py` | Periodic Q1 homogenisation, flux post-processing, analytical bounds, test geometries |
| `src/homog/design.py` | Inverse design: periodic filter, projection, D4 symmetry, MMA |
| `scripts/run_homog.py` | Verification, optimal cells, material-law study → `results/`, `figures/` |
| `scripts/make_animation.py` | Movie of the design evolving and heat flowing through it |
| `src/homog/stokes.py` | Periodic Stokes cell problem (permeability) and the walled-channel check |
| `src/homog/heat.py` | Convective cell problems (h_v, k_f,eff, K_mix), resolved strip and 1-D homogenised model |
| `scripts/run_permeability.py` | 2b heat-vs-flow study → `results/permeability_tradeoff.json` |
| `scripts/run_convection.py` | 2c verification, cell values, full-resolution comparison → `results/convection.json` |
| `src/homog/graded.py` | 2d plate model (Darcy + two-temperature heat), adjoint gradients, filter, full-resolution plate |
| `scripts/run_graded.py` | 2d cell table, uniform scan, SLSQP graded design → `results/graded.json` |
| `scripts/run_graded_check.py` | 2d full-resolution check of a design (server) → `results/graded_check/` |

**Tests (`tests/`, 24 passing):**
- homogeneous material returns itself, and laminates are exact;
- the tensor is symmetric positive-definite and lies within the Voigt/Reuss bounds;
- the checkerboard converges toward Dykhne, and discs lie within the HS bounds;
- sensitivities match finite differences;
- symmetrisation is a projection, and the filter wraps periodically;
- the optimiser beats a circular hole and exceeds 90% of the bound;
- local and global flux and energy balances hold to round-off;
- convection: plates Nu and Taylor–Aris, zero-flow limit equals the conduction FEM, the resolved strip matches the
  homogenised model and closes its energy balance, and walls cost a constant strip of flow;
- graded plate: adjoint gradients match finite differences, flow rate and energy balance of the plate model, uniform
  pressure drop equals Darcy, filter normalisation, and the resolved plate closes mass, power and energy.

## Run

```bash
pip install -r requirements.txt imageio-ffmpeg
python -m pytest -q tests          # 24 tests, ~30 s
python scripts/run_homog.py        # ~2 min
python scripts/make_animation.py   # ~1 min
python scripts/run_convection.py   # 2c, ~2 min
python scripts/run_graded.py       # 2d design, ~1 min
python scripts/run_graded_check.py opt 64   # 2d full-resolution check, ~30 min, ~11 GB
```
