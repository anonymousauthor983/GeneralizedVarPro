# Sparse mixture learning from a sketch

A mixture of Gaussians is learned from a compressive sketch of the data
(random Fourier moments) instead of the data themselves. The model is
over-parametrized with `L` candidate components: the component means `θ` are
the nonlinear parameters and the nonnegative masses `w` are the sparse linear
code, fitted under an ℓ1 + ridge penalty. GVP (called VarPro in the code)
solves the mass subproblem to numerical convergence for every candidate mean
vector and updates the means with the envelope gradient.

## Setting

- Data: `N = 20,000` samples from a mixture of `K = 5` Gaussians in R², known
  covariance `σ² I` with `σ = 0.15`, weights (0.40, 0.18, 0.16, 0.14, 0.12).
- Sketch: `M = 128` random Fourier features, frequencies `ω_j ~ N(0, 9 I)`,
  fixed during optimization. The optimizer only accesses the sketch.
- Model: `L = 36` candidate means initialized on a perturbed 6×6 grid,
  `λ = 0.008`, `μ = 1e-4`, 300 mean updates.
- Methods, all from identical initial means and optimally fitted initial
  masses: GVP (`varpro`), joint proximal gradient (`joint`), PALM (`palm`), and
  alternating minimization with 1 or 10 inner proximal steps (`prox1`,
  `prox10`). A control without the ℓ1 penalty and a convex fixed-grid
  reference are also run.

## Protocol and configuration selection

The configuration was selected among ten candidates using pilot seeds 0–4
only, then frozen before running the 20 evaluation seeds 201–220. All
candidates and their pilot outcomes are archived in
`reproducibility/exploration/`, including configurations in which every
method recovers five components; the selection criterion, the freeze time and
the hash of the frozen configuration are in
`reproducibility/exploration/selection.json`. All 20 evaluation seeds are run
once and reported; none is removed or re-run. This is an illustrative setting
chosen on pilot evidence, not a benchmark representative of all mixture
geometries.

## Results (20 paired trials, seeds 201–220)

| Method | Exactly five active components | Median ISE (×10⁻³) |
|---|---:|---:|
| GVP (`varpro`) | 18/20 | 1.50 |
| Joint proximal gradient | 0/20 | 1.52 |
| PALM | 0/20 | 1.54 |
| Alternating, 1 inner step | 0/20 | 1.52 |
| Alternating, 10 inner steps | 12/20 | 1.51 |

In the 18 compact GVP fits, the five estimated means match the true means
one-to-one within distance 0.020. The baselines also locate the mixture modes
but typically keep a redundant sixth component; density accuracy (ISE) is
comparable across methods. The control without sparsity penalty reaches a
median ISE of 0.60 × 10⁻³ with 16 active components.

## Reproduction

```bash
pip install -r requirements.txt
python reproducibility/experiment.py --out reproducibility/results   # all methods, seeds 201-220
python reproducibility/validate.py                                   # numerical checks (KKT, gradients, ISE)
python reproducibility/make_plots.py                                 # convergence figures -> figures/
python reproducibility/export_contours.py                            # contour paths -> figures/tikz/
bash build_contours.sh                                               # TikZ contour figures (pdfLaTeX + pgfplots)
```

`experiment.py` reads `reproducibility/frozen_config.json`. The pilot search
can be rerun with `reproducibility/explore.py` then
`reproducibility/explore_more.py` (seeds 0–4 only).

## Saved data

- `reproducibility/results/seed_*.npz`: frequencies, sketches, initializations,
  trajectories of means and masses, objectives, ISE, KKT residuals and timings
  for all methods.
- `summary.json` (per-trial results), `aggregate.json` (medians and
  quartiles), `validation.json` (numerical checks, matching of means,
  sensitivity of the activity threshold), `environment.json`.

ISE is computed analytically and checked by independent quadrature; joint and
reduced gradients are checked by finite differences. Active counts are
identical for activity thresholds from 1e-8 to 1e-3. Masses are nonnegative
but are not constrained to sum to one; they are normalized only to evaluate
or plot the density. Timings are wall-clock, single-threaded BLAS, and
machine-dependent; no conclusion relies on fine timing differences.
