import torch
import numpy as np
from core import *
from metrics_ops import cartesian_mask

def check(power,device='cpu'):
    torch.set_num_threads(2);torch.manual_seed(5)
    raw=torch.randn(1,6,16,16,device=device,dtype=torch.float64)
    x=torch.randn(1,1,16,16,device=device,dtype=torch.float64);m=cartesian_mask(16,4,0,device);y=acquisition(x,m)
    direction=torch.randn_like(raw);direction/=direction.norm();lam=.008 if power==.5 else .08
    def evaluate(v,diff):
        B=v/v.square().sum((-2,-1),keepdim=True).sqrt();H,b=gram(B,y,m);w,st=solve_codes(H,b,lam,power=power)
        if diff:w=implicit_codes(H,b,w,lam,st,power=power)
        return .5*(synth(B,w)-x).square().sum(),st
    raw.requires_grad_(True);loss,st=evaluate(raw,True);loss.backward();analytic=float((raw.grad*direction).sum());vals=[]
    for sign in [-1,1]:
        with torch.no_grad():v,_=evaluate(raw+sign*1e-5*direction,False);vals.append(float(v))
    fd=(vals[1]-vals[0])/2e-5;assert abs(fd-analytic)<1e-6,(power,analytic,fd)
    # Independent scalar global candidates for the half penalty, via cubic roots.
    if power==.5:
        for h in [.2,1.,3.]:
            for l in [.01,.3,2.]:
                for r in [-5.,-.1,0.,.1,5.]:
                    roots=np.roots([h,0.,-abs(r),l/2]);candidates=[0.]+[np.sign(r)*z.real**2 for z in roots if abs(z.imag)<1e-8 and z.real>0]
                    f=lambda w:.5*h*w*w-r*w+l*np.sqrt(abs(w));ours=half_coordinate(r,h,l)
                    assert abs(f(ours)-min(f(w) for w in candidates))<1e-9
    return dict(passed=True,power=power,analytic=analytic,finite_difference=fd,states=st)
if __name__=='__main__':
 import json
 for power in [1.,2.,.5]:print(json.dumps(check(power)))
