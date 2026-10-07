import json,hashlib
from pathlib import Path
import numpy as np
import torch
from check_priority_half import main
from core import solve_codes,half_diagnostics,gram,acquisition
r=main()
# Regression: the public solver must not accept a coordinate-stationary saddle.
h=.001*np.eye(4)+.999*np.ones((4,4));w=np.full(4,4.);b=h@w+.25
assert half_diagnostics(h,b,w,1.)['active_eigen_ratio']<0
z,st=solve_codes(torch.tensor(h)[None],torch.tensor(b)[None],1.,torch.tensor(w)[None],power=.5)
assert st[0]['active_eigen_ratio']>=0 and max(st[0]['active_stationarity'],st[0]['coordinate_residual'])<=1e-8
f=lambda q:.5*q@h@q-b@q+np.sqrt(np.abs(q)).sum()
assert f(z[0].numpy())<f(w)
# Exact mu=0 assembly: no data-fidelity contribution in joint training.
torch.manual_seed(17);B=torch.randn(2,4,8,8,dtype=torch.float64,requires_grad=True);x=torch.randn(2,1,8,8,dtype=torch.float64);mask=torch.ones(8,8);mask[:,1::2]=0;y=acquisition(x,mask)
H,b=gram(B,y,mask,x,0.);P=B.flatten(2)
assert torch.allclose(H,P@P.transpose(1,2),atol=1e-12)
assert torch.allclose(b,(P@x.flatten(2).transpose(1,2)).squeeze(-1),atol=1e-12)
H2,b2=gram(B,3*y,mask,x,0.);assert torch.equal(H,H2) and torch.equal(b,b2)
r.update(public_half_solver_saddle_regression=True,mu0_assembly=True,core_sha256=hashlib.sha256(Path(__file__).with_name('core.py').read_bytes()).hexdigest())
Path('PREFLIGHT.json').write_text(json.dumps(r,indent=2));print('PREFLIGHT PASSED',flush=True)
