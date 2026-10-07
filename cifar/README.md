# CIFAR-10: group-sparse readout of a ResNet-18

A ResNet-18 (CIFAR stem: 3x3 first convolution, no max-pooling) is trained on
CIFAR-10 with a linear readout `W` (512 x 10) and a squared loss on one-hot
targets. The readout carries the group penalty

    lambda * sum_j ||W_j||_2

over its rows. A zero row leaves feature `j` unused, and with it output channel
`j` of the last convolution, so readout sparsity removes parameters from the
network.

## Methods

| Name | Description |
|---|---|
| `gvp_dense` | GVP with a ridge readout only (no sparsity) |
| `joint_dense` | backbone and readout trained jointly with Adam |
| `gvp` | GVP: the group-sparse readout is solved at every step (warm-started FISTA, no gradient through the solver), the backbone is updated with the envelope gradient |
| `joint_prox` | Adam step on all parameters, then the group proximal operator on `W` with threshold `lr * lambda` |
| `prune_ft` | from `joint_dense`, keep the `s` largest readout rows (`s` = number kept by GVP at the same `lambda`), fine-tune 10 epochs |

All methods use Adam (lr 1e-3), batch size 8192, 50 epochs and ridge 1e-4.
For evaluation, batch normalization is in inference mode, and every method's
readout is re-solved without the sparsity penalty on its retained features
(`test_acc_refit`), so all methods share the same final refit.

## Usage

```bash
pip install -r ../requirements.txt

# one seed, all methods and penalty levels (CIFAR-10 is downloaded to ./data)
python cifar_group_sparse.py --seeds 0 --lams 1e-3 3e-3 1e-2 3e-2

# quick smoke test on random data (CPU is fine)
python cifar_group_sparse.py --fake --n_train 256 --n_test 128 --batch 128 --epochs 1 --ft_epochs 1

# full reproduction (3 seeds + full-precision check) and the paper figure
bash scripts/run_all.sh
```

Useful options: `--methods` (subset of methods), `--fp32` (disable mixed
precision; the default runs the backbone in fp16 so that batch size 8192 fits on
a 32 GB GPU), `--inner_iter` (FISTA budget), `--out` (results folder).

## Results

`results/mixed_precision/` holds the runs reported in the paper (3 seeds,
`final_seed*.csv`, aggregated in `summary.csv`); `results/full_precision/`
holds the full-precision rerun of the dense references and GVP, which matches
the mixed-precision results within one standard deviation.

Test accuracy after readout refit (%, mean ± std over 3 seeds):

| lambda | Features kept | GVP | Prune + fine-tune | Joint proximal |
|---|---|---|---|---|
| 1e-3 | 55 | 73.5 ± 0.6 | **79.2 ± 0.7** | 79.3 ± 0.2 (512 kept) |
| 3e-3 | 12 | 74.7 ± 0.7 | **77.5 ± 0.6** | 79.4 ± 0.6 (512 kept) |
| 1e-2 | 10 | 76.2 ± 0.4 | **77.0 ± 1.0** | 79.6 ± 0.2 (512 kept) |
| 3e-2 | 10 | **78.2 ± 0.7** | 76.6 ± 0.8 | 79.6 ± 0.1 (512 kept) |

Dense references (512 features): joint training 79.6 ± 0.4, GVP 73.5 ± 0.2.
With 10 retained features, 20.8% of the network parameters are removed.

`python plot_results.py` regenerates the figure from the CSVs.
