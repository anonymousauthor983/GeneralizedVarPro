"""Exact requested objective; positive weighted differences as conditional analysis.

No reference image and no extra regularizer enter the reduced objective.
Numerical VarPro uses the envelope derivative in the weights g, not abs'(D w).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def forward(w, mask):
    return torch.fft.fft2(w, norm='ortho') * mask


def adjoint(y, mask):
    return torch.fft.ifft2(y * mask, norm='ortho').real


def differences(w):
    return torch.cat((torch.roll(w, -1, -1)-w,
                      torch.roll(w, -1, -2)-w), dim=1)


def divergence(p):
    return (torch.roll(p[:, 0:1], 1, -1)-p[:, 0:1]
            + torch.roll(p[:, 1:2], 1, -2)-p[:, 1:2])


def objective(w, y, mask, g, lam):
    """One objective per image, divided by N (same scale for BOTH terms)."""
    n = w.shape[-1]*w.shape[-2]
    data = (forward(w, mask)-y).abs().square().flatten(1).sum(1)
    prior = (g*differences(w).abs()).flatten(1).sum(1)
    return (data + lam*prior)/n


def cartesian_mask(n, acceleration, seed, device='cpu', dtype=torch.float64):
    """Conjugate symmetric mask, DC included, exactly n//R lines when even."""
    count = n//acceleration
    if n % 2 or count % 2 or count < 4:
        raise ValueError('Even image size and even sample count >=4 required')
    gen = torch.Generator().manual_seed(seed)
    pairs = (count-2)//2
    low = min(pairs, max(1, round(n*(.08 if acceleration==4 else .04)/2)))
    remaining = torch.arange(low+1, n//2)
    chosen = remaining[torch.randperm(len(remaining), generator=gen)[:pairs-low]]
    positive = torch.cat((torch.arange(1, low+1), chosen))
    columns = torch.cat((torch.tensor([0, n//2]), positive, n-positive))
    mask = torch.zeros(n, n, dtype=dtype)
    mask[:, columns] = 1
    assert int(mask[0].sum()) == count
    return mask.to(device)


class WeightNet(nn.Module):
    """2 smooth weight maps, 16 channels, 2 residual blocks, no learned readout."""
    def __init__(self, conditional=True, channels=16):
        super().__init__()
        self.conditional = conditional
        self.head = nn.Conv2d(4, channels, 3, padding=1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1)) for _ in range(2)])
        self.tail = nn.Conv2d(channels, 2, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, y, mask):
        x = torch.fft.ifft2(y*mask, norm='ortho')
        size = max(8, y.shape[-1]//4)
        images = torch.cat((x.real, x.imag), dim=1)
        if not self.conditional:
            images = images*0
        images = F.interpolate(images, size=(size, size), mode='bilinear', align_corners=False)
        axis = torch.linspace(-1, 1, size, device=y.device, dtype=x.real.dtype)
        coords = torch.stack((axis[None, :].expand(size, size),
                              axis[:, None].expand(size, size)))[None]
        u = self.head(torch.cat((images, coords.expand(len(y), -1, -1, -1)), dim=1))
        for block in self.blocks:
            u = u+block(u)
        raw = F.interpolate(self.tail(u), size=y.shape[-2:], mode='bilinear', align_corners=False)
        # A priori bounds after normalization: 0.15/1.85 <= g <= 1.85/0.15.
        positive = .15+1.70*torch.sigmoid(raw)
        return positive/positive.mean(dim=(1, 2, 3), keepdim=True)


def kkt(w, p, y, mask, g, lam):
    """Stationarity, box feasibility, and complementarity; all per image.

    These are residuals, not a certified error bound on w without strong convexity.
    """
    dw = differences(w)
    residual = 2*adjoint(forward(w, mask)-y, mask)+divergence(p)
    scale = (2*adjoint(y, mask)).flatten(1).norm(dim=1).clamp_min(1e-12)
    stationarity = residual.flatten(1).norm(dim=1)/scale
    comp = (lam*g*dw.abs()-p*dw).flatten(1).sum(1).abs()
    # Scale by the full unnormalised objective; no hidden per-pixel tolerance.
    primal = objective(w, y, mask, g, lam)*(w.shape[-1]*w.shape[-2])
    complementarity = comp/primal.clamp_min(1e-12)
    feasibility = (p.abs()-lam*g).clamp_min(0).flatten(1).max(1)[0]
    return dict(stationarity=stationarity, complementarity=complementarity,
                feasibility=feasibility)


def solve(y, mask, g, lam, tol=1e-4, max_iter=12000, iterations=None, initial=None):
    """ADMM, splitting v=Dw. FFT-exact quadratic w step, exact soft threshold v.

    No ridge and no L1 smoothing. The algorithmic rho is NOT a penalty added
    to the requested objective. Each finite run is still a numerical solve.
    Set iterations=64 to intentionally truncate/unroll; otherwise enforce KKT.
    """
    rho=max(10*float(lam),.01)
    w = adjoint(y, mask) if initial is None else initial[0]
    p = torch.zeros_like(differences(w)) if initial is None else initial[1]
    # Warm dual values must remain feasible when g changes.
    p = torch.minimum(torch.maximum(p, -lam*g), lam*g)
    u=p/rho
    v=differences(w)
    n = w.shape[-1]
    reverse = (-torch.arange(n, device=w.device)) % n
    symmetric = .5*(mask+mask.index_select(-2, reverse).index_select(-1, reverse))
    rhs = 2*torch.fft.fft2(adjoint(y, mask), norm='ortho')
    axis=torch.arange(n,device=w.device,dtype=w.dtype)
    eigen=4*torch.sin(math.pi*axis/n).square()
    laplacian=eigen[:,None]+eigen[None,:]
    denom = 2*symmetric+rho*laplacian
    if float(denom.min())<=0:raise ValueError('DC must be sampled for this exact FFT step')
    budget = max_iter if iterations is None else iterations
    for it in range(1, budget+1):
        w=torch.fft.ifft2((rhs+rho*torch.fft.fft2(divergence(v-u),norm='ortho'))/denom,norm='ortho').real
        relaxed=1.6*differences(w)-.6*v
        shifted=relaxed+u
        v=shifted.sign()*(shifted.abs()-lam*g/rho).clamp_min(0)
        u=u+relaxed-v
        p=rho*u
        if iterations is None and it % 50 == 0:
            status = kkt(w, p, y, mask, g, lam)
            if max(v.max().item() for v in status.values()) <= tol:
                break
    status = {key:float(val.max().detach()) for key,val in kkt(w,p,y,mask,g,lam).items()}
    status.update(iterations=it, converged=max(status.values()) <= tol)
    if iterations is None and not status['converged']:
        raise RuntimeError('Inner solve failed requested accuracy: '+str(status))
    return w, p, status


def image_metrics(w, x):
    """Signed-image fidelity, magnitude PSNR, local Gaussian-window SSIM.

    Fixed display range 4.5 for compatibility with prepared RMS-normalised data.
    """
    mag = w.abs()
    mse = (mag-x).square().flatten(1).mean(1)
    t = torch.arange(11, device=x.device, dtype=x.dtype)-5
    kernel = torch.exp(-t.square()/4.5);kernel = kernel/kernel.sum()
    filt = (kernel[:,None]*kernel[None,:])[None,None]
    conv = lambda a:F.conv2d(a,filt)
    ux,uz = conv(x),conv(mag)
    vx,vz,cov = conv(x*x)-ux*ux,conv(mag*mag)-uz*uz,conv(x*mag)-ux*uz
    c1,c2 = (.01*4.5)**2,(.03*4.5)**2
    ssim = (((2*ux*uz+c1)*(2*cov+c2))/((ux*ux+uz*uz+c1)*(vx+vz+c2))).flatten(1).mean(1)
    return dict(psnr=10*math.log10(4.5**2)-10*mse.clamp_min(1e-30).log10(),
                ssim=ssim,nmse=(w-x).square().flatten(1).sum(1)/x.square().flatten(1).sum(1),
                signed_mse=(w-x).square().flatten(1).mean(1))
