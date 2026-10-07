"""Reproduce the CIFAR-10 figure of the paper (accuracy vs. penalty level).

Reads the per-seed CSVs written by cifar_group_sparse.py and plots the test
accuracy after readout refit of GVP and of pruning + fine-tuning, with the
dense references as horizontal lines.

Example:
  python plot_results.py --results results/mixed_precision --out fig_cifar_group_sparse.pdf
  python plot_results.py --usetex          # LaTeX fonts (requires a LaTeX install)
"""

import argparse
import glob
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/mixed_precision")
    ap.add_argument("--out", default="fig_cifar_group_sparse.pdf")
    ap.add_argument("--usetex", action="store_true")
    args = ap.parse_args()

    plt.rcParams.update({
        "text.usetex": args.usetex, "font.family": "serif",
        "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    })
    if args.usetex:
        plt.rcParams["font.serif"] = ["Computer Modern Roman"]

    d = pd.concat(pd.read_csv(f) for f in glob.glob(os.path.join(args.results, "final_seed*.csv")))
    g = d.groupby(["lam", "method"]).agg(acc=("test_acc_refit", "mean"), sd=("test_acc_refit", "std"),
                                          act=("active", "mean")).reset_index()
    lams = sorted(g[g.method == "gvp"].lam.unique())
    col = lambda m, k: np.array([g[(g.lam == l) & (g.method == m)][k].item() for l in lams])
    dense_joint = g[g.method == "joint_dense"].acc.item()
    dense_gvp = g[g.method == "gvp_dense"].acc.item()
    blue, orange = "#1f5fa8", "#c8501a"

    fig, ax = plt.subplots(figsize=(4.0, 2.75))
    x = np.arange(len(lams))
    ax.axhline(dense_joint, color="0.25", lw=0.9, ls="--", label="Dense joint (512 features)")
    ax.axhline(dense_gvp, color=blue, lw=0.9, ls=":", label="Dense GVP (512 features)")
    methods = [("gvp", blue, "o", "GVP")]
    if (g.method == "prune_ft").any():
        methods.append(("prune_ft", orange, "s", "Prune + fine-tune"))
    for m, c, mk, lab in methods:
        ax.errorbar(x, col(m, "acc"), yerr=col(m, "sd"), color=c, lw=1.3, marker=mk, ms=4,
                    capsize=2.5, capthick=0.8, elinewidth=0.8, label=lab, zorder=3)
    ax.set_xticks(x, [f"{l:g}\n({round(a)})" for l, a in zip(lams, col("gvp", "act"))])
    ax.set_xlabel(r"Penalty $\lambda$ (features retained)")
    ax.set_ylabel("Test accuracy (%)" if not args.usetex else r"Test accuracy (\%)")
    ax.grid(axis="y", color="0.9", lw=0.5)
    ax.set_axisbelow(True)
    h, l = ax.get_legend_handles_labels()
    order = list(range(2, len(h))) + [0, 1]
    ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
              bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=False)
    fig.tight_layout(pad=0.3)
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
