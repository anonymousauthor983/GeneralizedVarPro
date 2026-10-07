import json,torch
from pathlib import Path
from core import diagnostics,implicit_codes
from check_powers import check
h=torch.eye(2,dtype=torch.float64)[None].requires_grad_(True)
b=torch.tensor([[2.,1.+5e-9]],dtype=torch.float64,requires_grad=True)
w=torch.tensor([[1.,0.]],dtype=torch.float64)
st=[diagnostics(h[0].detach().numpy(),b[0].detach().numpy(),w[0].numpy(),1.)]
assert st[0]['kkt']<1e-8 and st[0]['inactive_margin']<0
z=implicit_codes(h,b,w,1.,st,power=1.)
assert torch.allclose(z,b-1,atol=1e-15,rtol=0)
g=torch.autograd.grad(z.sum(),b)[0];assert torch.equal(g,torch.ones_like(b))
boundary=torch.tensor([[2.,1.]],dtype=torch.float64)
w=torch.tensor([[1.,0.]],dtype=torch.float64);st=[diagnostics(h[0].detach().numpy(),boundary[0].numpy(),w[0].numpy(),1.)]
try:implicit_codes(h,boundary,w,1.,st,power=1.)
except RuntimeError as e:assert 'Support transition' in str(e)
else:raise AssertionError('Exact kink must still be rejected')
result=dict(passed=True,weak_violation_refined=True,known_derivative_checked=True,true_transition_still_rejected=True,gradients=[check(p) for p in [1.,2.,.5]])
Path('CHECK_MARGIN.json').write_text(json.dumps(result,indent=2));print('MARGIN CHECK PASSED',flush=True)
