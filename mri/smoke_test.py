"""Synthetic checks only: no data and no optimizer/training steps."""
import torch
import json
from core import FeatureNet,acquisition,features_input,gram,synth,solve_codes,implicit_codes
from metrics_ops import cartesian_mask,solve,differences,image_metrics
from pathlib import Path
torch.set_num_threads(2);torch.manual_seed(7)
x=torch.rand(1,1,32,32,dtype=torch.float64);mask=cartesian_mask(32,4,100,'cpu');y=acquisition(x,mask)
net=FeatureNet();B=net(features_input(y));assert B.shape==(1,16,32,32)
assert torch.allclose(B.square().sum((-2,-1)),torch.ones(1,16,dtype=torch.float64),atol=1e-10)
H,b=gram(B,y,mask);w,st=solve_codes(H,b,.01,power=2);used=implicit_codes(H,b,w,.01,st,power=2)
loss=.5*(synth(B,used)-x).square().mean();loss.backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
rec,_,tv=solve(y,mask,torch.ones_like(differences(x)),.04,tol=1e-5,max_iter=240000)
assert tv['converged'];assert all(torch.isfinite(v).all() for v in image_metrics(rec,x).values())
Path('SMOKE_TEST.json').write_text(json.dumps(dict(passed=True,network_normalization=True,network_backward=True,tv=tv),indent=2))
print('SMOKE TEST PASSED')
