#!/bin/bash
# Reproduces the CIFAR-10 results of the paper (one GPU per seed recommended).
# Each seed runs all methods for the four penalty levels (~50 min on one V100).
set -e
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
    python cifar_group_sparse.py --seeds $seed \
        --lams 1e-3 3e-3 1e-2 3e-2 --epochs 50 --batch 8192 \
        --out results/mixed_precision
done

# Full-precision check of the dense references and GVP (needs >= 40 GB of GPU memory).
for seed in 0 1 2; do
    python cifar_group_sparse.py --seeds $seed \
        --lams 1e-3 3e-3 1e-2 3e-2 --epochs 50 --batch 8192 \
        --fp32 --methods gvp_dense joint_dense gvp \
        --out results/full_precision
done

python plot_results.py --results results/mixed_precision --out fig_cifar_group_sparse.pdf
