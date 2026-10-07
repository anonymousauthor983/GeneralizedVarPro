import json
import torch
import numpy as np
from core import half_active_refine,half_diagnostics
from check_powers import check

def main():
    records=[]
    for n in [2,4,8,16]:
        for rho in [.99,.999,.9999]:
            h=(1-rho)*np.eye(n)+rho*np.ones((n,n))
            w=np.full(n,4.);b=h@w+.25;old=w.copy()
            value=lambda v:float(.5*v@h@v-b@v+np.sqrt(np.abs(v)).sum())
            before=half_diagnostics(h,b,w,1.)
            assert before['active_eigen_ratio']<0
            st,steps=half_active_refine(h,b,w,1.,1e-8)
            assert value(w)<value(old)
            assert max(st['active_stationarity'],st['coordinate_residual'])<=1e-8
            assert st['active_eigen_ratio']>0
            records.append(dict(n=n,rho=rho,steps=steps,status=st))
    result=dict(passed=True,curvature_cases=records,gradients=[check(p,'cpu') for p in [1.,2.,.5]])
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
