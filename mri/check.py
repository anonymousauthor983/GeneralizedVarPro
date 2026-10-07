import json
import torch
from core import *
from metrics_ops import cartesian_mask

def check(device='cpu'):
    torch.set_num_threads(2);torch.manual_seed(5)
    dtype=torch.float64
    raw=torch.randn(1,6,16,16,device=device,dtype=dtype)
    x=torch.randn(1,1,16,16,device=device,dtype=dtype)
    mask=cartesian_mask(16,4,0,device);y=acquisition(x,mask)
    direction=torch.randn_like(raw);direction/=direction.norm()
    lam=.08;records=[]
    def value(r,differentiate):
        B=r/r.square().sum((-2,-1),keepdim=True).sqrt();H,b=gram(B,y,mask)
        w,st=solve_codes(H,b,lam)
        if differentiate:w=implicit_codes(H,b,w,lam,st)
        return (synth(B,w)-x).square().sum()/2,st
    raw.requires_grad_(True);f,st=value(raw,True);f.backward();analytic=float((raw.grad*direction).sum());values=[]
    for sign in [-1,1]:
        with torch.no_grad():v,_=value(raw+sign*1e-5*direction,False);values.append(float(v))
    fd=(values[1]-values[0])/2e-5;assert abs(fd-analytic)<1e-6,(fd,analytic)
    # Identity Hessian Lasso has analytic soft threshold.
    H=torch.eye(6,device=device,dtype=dtype)[None];b=torch.tensor([[1.,-.5,.01,0.,2.,-.1]],device=device,dtype=dtype)
    w,st=solve_codes(H,b,lam);exact=b.sign()*(b.abs()-lam).clamp_min(0);assert torch.allclose(w,exact,atol=1e-12)
    # Partial gradient for the joint quadratic agrees with autograd.
    B=raw.detach()/raw.detach().square().sum((-2,-1),keepdim=True).sqrt();hj,bj=gram(B,y,mask,x,mu=1.)
    ww=torch.randn(1,6,device=device,dtype=dtype,requires_grad=True);r=synth(B,ww)
    loss=.5*(r-x).square().sum()+.5*(acquisition(r,mask)-y).abs().square().sum();loss.backward()
    assert torch.allclose(ww.grad,torch.einsum('bij,bj->bi',hj,ww)-bj,atol=1e-10)
    return dict(passed=True,implicit_gradient=analytic,finite_difference=fd,error=abs(fd-analytic))
if __name__=='__main__':print(json.dumps(check(),indent=2))
