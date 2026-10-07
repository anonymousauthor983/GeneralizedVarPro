"""CIFAR-10 / ResNet-18: group-sparse readout that removes last-layer channels.

The readout W (512 x 10) is penalized per feature (row of W):
    lambda * sum_j ||W_j||_2
A zero row removes feature j, i.e. output channel j of the last convolution of
ResNet-18 (layer4[1].conv2 and its BatchNorm), so the sparsity is a real
parameter reduction, reported as `params_removed`.

Methods (same backbone, batch, epochs, Adam lr, MSE on one-hot targets):
  gvp_dense      readout solved exactly at every step (ridge only)
  gvp            readout solved exactly at every step with the group penalty
                 (FISTA, warm start, no grad), envelope gradient for the backbone
  joint_dense    readout trained jointly (Adam)
  joint_prox     joint Adam, then group prox on the readout with threshold lr*lambda
  prune_ft       joint_dense, keep the s features with the largest readout rows
                 (s = number kept by gvp at the same lambda), fine-tune 10 epochs

Final evaluation, identical for all methods (BatchNorm in eval mode):
  test_acc        the method's own readout (for gvp: the group problem re-solved
                  on the full training set at the final backbone)
  test_acc_refit  readout re-solved without the sparsity penalty on the kept
                  features only (same courtesy for every method)

Example:
  python cifar_group_sparse.py --seeds 0 --lams 1e-3 3e-3 1e-2 3e-2
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
AMP = device == "cuda"  # set to False by --fp32
FEAT = 512
N_CLASSES = 10


def sync():
    if device == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------------
def load_cifar10(n_train, n_test, fake=False, data_dir="data"):
    if fake:
        g = torch.Generator().manual_seed(0)
        X_tr = torch.randn(n_train, 3, 32, 32, generator=g)
        X_te = torch.randn(n_test, 3, 32, 32, generator=g)
        y_tr = torch.randint(0, 10, (n_train,), generator=g)
        y_te = torch.randint(0, 10, (n_test,), generator=g)
    else:
        import torchvision
        import torchvision.transforms as T
        tfm = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465),
                                                   (0.2470, 0.2435, 0.2616))])
        tr = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
        te = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tfm)
        rng = np.random.default_rng(0)
        i_tr = rng.choice(len(tr), n_train, replace=False)
        i_te = rng.choice(len(te), n_test, replace=False)
        X_tr = torch.stack([tr[i][0] for i in i_tr]); y_tr = torch.tensor([tr[i][1] for i in i_tr])
        X_te = torch.stack([te[i][0] for i in i_te]); y_te = torch.tensor([te[i][1] for i in i_te])
    Y_tr = nn.functional.one_hot(y_tr, N_CLASSES).float()
    return X_tr.to(device), Y_tr.to(device), y_tr.to(device), X_te.to(device), y_te.to(device)


def make_backbone():
    import torchvision.models as tvm
    net = tvm.resnet18(weights=None)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    net.fc = nn.Identity()
    return net


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = make_backbone()
        self.W = nn.Parameter(torch.zeros(FEAT, N_CLASSES))
        self.b = nn.Parameter(torch.zeros(N_CLASSES))

    def forward(self, X):
        return self.phi(X) @ self.W + self.b


def last_conv_params_per_channel(net):
    """Parameters attached to one output channel of layer4[1].conv2 (+BN)."""
    conv = net.phi.layer4[1].conv2
    return conv.weight[0].numel() + 2  # kernel (512*3*3) + BN gamma/beta


def total_params(net):
    return sum(p.numel() for p in net.phi.parameters()) + FEAT * N_CLASSES + N_CLASSES


# ---------------------------------------------------------------------------
# Group-lasso readout: min 1/(2n)||HW + 1b - Y||^2 + ridge/2||W||^2 + lam sum_j ||W_j||
# ---------------------------------------------------------------------------
def group_prox(W, tau):
    n = W.norm(dim=1)
    return W * (1.0 - tau / n.clamp_min(1e-30)).clamp_min(0.0)[:, None]


@torch.no_grad()
def solve_from_stats(G, R, h_mean, y_mean, lam, ridge, W0=None, max_iter=100, tol=1e-6):
    L = torch.linalg.eigvalsh(G)[-1].item() + ridge
    step = 1.0 / L
    W = torch.zeros(G.shape[0], R.shape[1], device=G.device) if W0 is None else W0.clone()
    Z, t = W.clone(), 1.0
    for _ in range(max_iter):
        W_new = group_prox(Z - step * (G @ Z - R + ridge * Z), step * lam)
        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        Z = W_new + ((t - 1.0) / t_new) * (W_new - W)
        done = ((W_new - W).norm() / W.norm().clamp_min(1.0)).item() <= tol
        W, t = W_new, t_new
        if done:
            break
    b = (y_mean - h_mean @ W).squeeze(0)
    return W, b


def batch_stats(H, Y):
    h_mean, y_mean = H.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Hc, Yc = H - h_mean, Y - y_mean
    return Hc.T @ Hc / H.shape[0], Hc.T @ Yc / H.shape[0], h_mean, y_mean


def features(net, X):
    """Backbone in fp16 on GPU (memory: batch 8192 must fit a 32 GB V100);
    everything downstream (readout solve, losses) stays in fp32."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=AMP):
        H = net.phi(X)
    return H.float()


@torch.no_grad()
def full_features(net, X, bs=2048):
    net.eval()
    H = torch.cat([features(net, X[i:i + bs]) for i in range(0, X.shape[0], bs)])
    net.train()
    return H


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def mse(pred, Y):
    return 0.5 * (pred - Y).pow(2).sum() / Y.shape[0]


def train(method, data, seed, lam, args, init_net=None, keep_s=None):
    X_tr, Y_tr, y_tr, X_te, y_te = data
    torch.manual_seed(seed)
    net = Net().to(device) if init_net is None else init_net
    is_gvp = method.startswith("gvp")
    params = list(net.phi.parameters()) if is_gvp else list(net.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    lam_eff = lam if method in ("gvp", "joint_prox") else 0.0
    epochs = args.ft_epochs if method == "prune_ft" else args.epochs

    mask = None
    if method == "prune_ft":
        with torch.no_grad():
            norms = net.W.norm(dim=1)
            mask = torch.zeros(FEAT, device=device)
            mask[norms.topk(keep_s).indices] = 1.0
            net.W.mul_(mask[:, None])

    t_solve = 0.0
    sync(); t0 = time.time()
    n = X_tr.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad(set_to_none=True)
            H = features(net, X_tr[idx])
            if is_gvp:
                sync(); ts = time.time()
                G, R, hm, ym = batch_stats(H.detach(), Y_tr[idx])
                W, b = solve_from_stats(G, R, hm, ym, lam_eff, args.ridge,
                                        W0=net.W.detach(), max_iter=args.inner_iter)
                with torch.no_grad():
                    net.W.copy_(W); net.b.copy_(b)
                sync(); t_solve += time.time() - ts
                loss = mse(H @ net.W.detach() + net.b.detach(), Y_tr[idx])
            else:
                loss = mse(H @ net.W + net.b, Y_tr[idx])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                if method == "joint_prox":
                    net.W.copy_(group_prox(net.W, args.lr * lam_eff))
                if mask is not None:
                    net.W.mul_(mask[:, None])
    sync()
    return net, time.time() - t0, t_solve


@torch.no_grad()
def evaluate(net, method, data, lam, args):
    X_tr, Y_tr, y_tr, X_te, y_te = data
    H_tr, H_te = full_features(net, X_tr), full_features(net, X_te)
    G, R, hm, ym = batch_stats(H_tr, Y_tr)
    if method == "gvp":
        W, b = solve_from_stats(G, R, hm, ym, lam, args.ridge, W0=net.W.detach(), max_iter=2000)
    else:
        W, b = net.W.detach(), net.b.detach()
    keep = W.norm(dim=1) > 1e-10
    s = int(keep.sum().item())
    acc = ((H_te @ W + b).argmax(1) == y_te).float().mean().item() * 100
    # refit: unpenalized (ridge) readout on the kept features only
    Gk, Rk = G[keep][:, keep], R[keep]
    Wk, bk = solve_from_stats(Gk, Rk, hm[:, keep], ym, 0.0, args.ridge, max_iter=3000)
    acc_refit = ((H_te[:, keep] @ Wk + bk).argmax(1) == y_te).float().mean().item() * 100
    per_ch = last_conv_params_per_channel(net)
    removed = (FEAT - s) * (per_ch + N_CLASSES)
    return dict(active=s, test_acc=acc, test_acc_refit=acc_refit,
                params_removed=removed, params_removed_pct=100.0 * removed / total_params(net))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--ft_epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--inner_iter", type=int, default=100)
    ap.add_argument("--n_train", type=int, default=50000)
    ap.add_argument("--n_test", type=int, default=10000)
    ap.add_argument("--fake", action="store_true", help="random data, for smoke tests")
    ap.add_argument("--data_dir", default="data", help="where CIFAR-10 is (or will be downloaded)")
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--fp32", action="store_true", help="disable mixed precision")
    ap.add_argument("--methods", nargs="+",
                    default=["gvp_dense", "joint_dense", "gvp", "joint_prox", "prune_ft"])
    args = ap.parse_args()
    global AMP
    if args.fp32:
        AMP = False
    os.makedirs(args.out, exist_ok=True)

    data = load_cifar10(args.n_train, args.n_test, fake=args.fake, data_dir=args.data_dir)
    print(f"device={device} amp={AMP} n_train={args.n_train} epochs={args.epochs} batch={args.batch}", flush=True)
    tag = "_".join(str(s) for s in args.seeds)
    path = os.path.join(args.out, f"final_seed{tag}.csv")
    fields = ["method", "seed", "lam", "active", "test_acc", "test_acc_refit",
              "params_removed", "params_removed_pct", "time_s", "solve_time_s"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()

        def log(method, seed, lam, net, t, ts):
            ev = evaluate(net, method, data, lam, args)
            row = dict(method=method, seed=seed, lam=lam, time_s=t, solve_time_s=ts, **ev)
            wr.writerow(row); f.flush()
            print(f"  {method:11s} lam={lam:<7g} active={ev['active']:3d} acc={ev['test_acc']:.2f} "
                  f"refit={ev['test_acc_refit']:.2f} removed={ev['params_removed_pct']:.1f}% "
                  f"t={t:.0f}s solve={ts:.0f}s", flush=True)
            return ev

        for seed in args.seeds:
            print(f"-- seed {seed}", flush=True)
            M = set(args.methods)
            if "gvp_dense" in M:
                net, t, ts = train("gvp_dense", data, seed, 0.0, args)
                log("gvp_dense", seed, 0.0, net, t, ts)
            if "joint_dense" in M or "prune_ft" in M:
                joint, t, ts = train("joint_dense", data, seed, 0.0, args)
                log("joint_dense", seed, 0.0, joint, t, ts)
                joint_state = {k: v.clone() for k, v in joint.state_dict().items()}
            for lam in args.lams:
                s = 0
                if "gvp" in M:
                    net, t, ts = train("gvp", data, seed, lam, args)
                    s = log("gvp", seed, lam, net, t, ts)["active"]
                if "joint_prox" in M:
                    net, t, ts = train("joint_prox", data, seed, lam, args)
                    log("joint_prox", seed, lam, net, t, ts)
                if "prune_ft" in M and 0 < s < FEAT:
                    ft = Net().to(device); ft.load_state_dict(joint_state)
                    net, t, ts = train("prune_ft", data, seed, lam, args, init_net=ft, keep_s=s)
                    log("prune_ft", seed, lam, net, t, ts)
                if device == "cuda":
                    torch.cuda.empty_cache()
    print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
