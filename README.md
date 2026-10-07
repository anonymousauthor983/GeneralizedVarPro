# Generalized Variable Projection

Code for the paper *Generalized Variable Projection for Training Statistical
Models With Sparsity* (anonymous submission).

Generalized Variable Projection (GVP) trains separable models
`y ≈ Φ_θ w` whose linear weights `w` should be sparse. At each iteration, the
weights are replaced by the solution of their regularized, possibly nonsmooth,
convex inner problem, and only the nonlinear parameters `θ` are updated, with
any first-order optimizer and without differentiating through the inner
solver.

## Repository structure

| Folder | Experiment |
|---|---|
| [`cifar/`](cifar/) | Group-sparse readout of a ResNet-18 on CIFAR-10 |
| `mri/` | MRI reconstruction with learnable features *(to be added)* |
| `sketching/` | Sketched spike deconvolution *(to be added)* |

Each folder is self-contained and has its own README with the exact commands
to reproduce the corresponding results of the paper.

## Installation

```bash
pip install -r requirements.txt
```

Python >= 3.10 and PyTorch >= 2.0. A GPU is recommended for the CIFAR-10
experiments.

## License

GNU Affero General Public License v3.0, see [`LICENSE`](LICENSE).
