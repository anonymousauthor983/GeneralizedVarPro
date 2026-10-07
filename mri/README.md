# MRI reconstruction with learnable synthesis features

## Requirements

Python 3.12, NVIDIA GPU and a CUDA-compatible driver for training. Historical environment: PyTorch 2.6.0 with CUDA 12.4, NumPy 1.26.4, h5py 3.16.0. See `environment_historical.json`.

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt -r requirements-figures.txt
python test.py
```

Alternatively:

```bash
docker build -t mri-reconstruction .
docker run --rm mri-reconstruction python test.py
```

## Data

Obtain authorized fastMRI knee HDF5 files containing `reconstruction_rss`. Raw data and model checkpoints are not distributed. `data_manifest.json` records the exact volumes, slices, normalization scales and prepared-array checksums.

```bash
python prepare_data.py --raw-root /path/to/fastmri --output data
```

The fixed split contains 1,920 training slices from 120 volumes and 480 validation slices from 30 disjoint volumes, using CORPD_FBK images cropped/padded to 320×320. Prepared arrays are verified against archived SHA-256 checksums. The code does not evaluate an independent test set.

Measurements are simulated by Cartesian Fourier undersampling of RSS reference images at ×4 and ×8, with no added noise. This is not reconstruction from raw multicoil measurements. Reference-based RMS normalization to sqrt(2) is a limitation of this simulated protocol. Historical radial-mask metadata in the manifest is not used: `metrics_ops.cartesian_mask` defines the experiment masks (seeds 0/100 for training/validation).

## Training

```bash
# Inspect the full schedule without running training.
python reproduce.py --data data --dry-run

# One configuration: seed 0, l1, ×8, intermediate lambda.
python reproduce.py --data data --seeds 0 --powers 1 --accelerations 8 --lambda-indices 1

# Full schedule: 66 runs per seed, five seeds.
python reproduce.py --data data --output runs
```

The launcher runs sequentially on one GPU, logs each run to `runs/seed*/mu*/logs/`, stops on failure and resumes from the last complete epoch. Completed results are checked before skipping. Concurrent launches using the same output directory are rejected. Use separate output directories for independent jobs.

For container execution, mount prepared data and a writable output directory:

```bash
docker run --rm --gpus all   -v "$PWD/data:/data:ro" -v "$PWD/runs:/runs"   mri-reconstruction python reproduce.py --data /data --output /runs
```

The architecture is a U-Net with two input channels (real/imaginary zero-filled reconstruction), widths 32/64/128/256 and a 512-channel bottleneck. It outputs 16 spatial feature maps, each normalized to unit l2 norm. Image-specific coefficients synthesize the reconstruction. Network parameters use float32; coefficient solves use float64.

Training uses eight epochs, batch size four, Adam with learning rate 0.001 and 3,840 updates. Dataset dimensions and iteration budgets in `run.py` implement this fixed protocol; editing only `base_config.json` does not change those constants.

For f(theta,w) = 1/2 ||Psi_theta(y)w-x||² and d(theta,w) = 1/2 ||A Psi_theta(y)w-y||²:

| Method | Training objective and update |
| --- | --- |
| `projected` (VarPro) | Solve d+lambda R for w; differentiate the supervised outer objective f through the regular solution branch. |
| `joint_prox` | Optimize f+mu d+lambda R with simultaneous network-gradient and coefficient-proximal updates. |
| `alternating` | Solve f+mu d+lambda R for w, then take a partial network-gradient step with w fixed. |

These methods have different training objectives; the comparison does not isolate optimizer choice alone. Inference always solves d+lambda R without ground-truth access, using the training lambda.

R(w)=sum |w_j|^p for p=1, 2 and 1/2. The p=2 penalty is the squared l2 norm. All families use mu=1; l1 also includes mu=0 for joint and alternating methods. VarPro is independent of mu and is reused. Lambda values are fixed across seeds in `calibration/`; fractions are 0.001/0.01/0.1 for p=1 and p=1/2, and 0.01/0.1/1 for p=2. `calibrate.py` retains the original calibration procedure, including pilot optimizer steps; it is not invoked by reproduction or tests.

Coefficient solves are numerical, with residual tolerance 1e-8. Differentiation rejects irregular support transitions and invalid curvature. The nonconvex p=1/2 solver provides locally checked solutions, not certified global minima. No general convergence guarantee follows from the included tests.

## Evaluation and figures

```bash
python evaluate_tv.py --data data --output tv_results.json
python summarize.py runs --output summary.json
python plot_reference_l1.py
python reconstruct_slice24.py --data data --runs runs --output figures
```

TV uses ||Az-y||² + lambda_TV ||grad z||1 with periodic anisotropic differences, ADMM tolerance 1e-5 and at most 240,000 iterations. Its grid is 0.01/0.04/0.16; the reference figure uses 0.04. TV and learned-method lambda values have different meanings.

Metrics: magnitude PSNR with fixed range 4.5, local Gaussian-window SSIM, signed-image NMSE, exact-zero percentage among 16 coefficients and active count at relative threshold 1e-6. `summarize.py` computes means, sample variances and standard deviations across seeds completed by all three methods in each configuration. A single seed has undefined variance. TV has no directly comparable coefficient-sparsity percentage.

`reference/` contains frozen manuscript statistics, not newly computed results. `plot_reference_l1.py` reproduces that frozen plot. `reconstruct_slice24.py` uses reproduced seed-0 checkpoints at intermediate lambda and mu=1 to render validation slice 24, both accelerations and absolute errors. Only load trusted checkpoints.

## Verification

The included GitHub Actions workflow assumes this directory is the repository root. In a monorepo, move the workflow to the root `.github/workflows/` and set `defaults.run.working-directory: mri`.

`python test.py` runs numerical regression tests, checks the 330-run schedule and validates the archived data split. It performs no training and needs no fastMRI data. `RELEASE_CHECKS.json` records release validation and its limits; `SHA256SUMS.json` records file checksums. GPU training and full raw-data preparation must be validated on the deployment system. Floating-point differences across devices and library versions can affect results.

Source attribution: fastMRI (NYU/Meta); MRI setup informed by arXiv:2011.04268. Dataset access remains subject to its terms. The code is released under the repository license (AGPL-3.0, see `../LICENSE`).
