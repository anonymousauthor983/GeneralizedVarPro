"""Paired synthetic experiment: sparse mixture estimation from a Fourier sketch.
Run: OPENBLAS_NUM_THREADS=1 python experiment.py --out results
No raw observations are used by the optimizers. All diagnostics use saved states.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse, json, platform, time
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import nnls

CFG=json.loads((Path(__file__).resolve().parent/'frozen_config.json').read_text())
METHODS=['joint','prox1','prox10','palm','varpro']

def atoms(theta,omega,cfg=CFG):
    return np.exp(1j*(omega@theta.T)-.5*cfg['sigma']**2*np.sum(omega**2,axis=1)[:,None])/np.sqrt(len(omega))

def value(A,w,z,cfg=CFG):
    r=A@w-z
    return .5*np.vdot(r,r).real+.5*cfg['mu']*np.dot(w,w)+cfg['lam']*w.sum()

def inner(A,z,cfg=CFG):
    # Exact QP reformulation, up to an irrelevant constant:
    # .5||[Re A; Im A; sqrt(mu) I]w-[Re z; Im z; -lambda/sqrt(mu) 1]||^2.
    p=A.shape[1];mu=cfg['mu']
    B=np.vstack([A.real,A.imag,np.sqrt(mu)*np.eye(p)])
    b=np.concatenate([z.real,z.imag,np.full(p,-cfg['lam']/np.sqrt(mu))])
    w=nnls(B,b,maxiter=30*p)[0]
    return w

def kkt(A,w,z,cfg=CFG):
    g=(A.conj().T@(A@w-z)).real+cfg['mu']*w+cfg['lam']
    return float(np.max(np.abs(np.where(w>1e-10,g,np.minimum(g,0)))))

def grad_theta(A,w,z,omega):
    return np.einsum('mk,md,m->kd',(-1j*A.conj())*w[None,:],omega,A@w-z).real

def ise(theta,w,cfg=CFG):
    # Exact L2 density error; normalize masses ONLY for this diagnostic.
    v=w/w.sum();ts=np.array(cfg['true_centers']);ps=np.array(cfg['true_weights'])
    def gram(a,b):return np.exp(-np.sum((a[:,None,:]-b[None,:,:])**2,axis=2)/(4*cfg['sigma']**2))/(4*np.pi*cfg['sigma']**2)
    return max(0.,float(v@gram(theta,theta)@v+ps@gram(ts,ts)@ps-2*v@gram(theta,ts)@ps))

def make_problem(seed,cfg=CFG):
    rng=np.random.default_rng(seed)
    omega=rng.normal(0,cfg['frequency_scale'],(cfg['m'],2))
    ts=np.array(cfg['true_centers']);ps=np.array(cfg['true_weights'])
    x=ts[rng.choice(len(ts),size=cfg['n'],p=ps)]+cfg['sigma']*rng.normal(size=(cfg['n'],2))
    start=time.perf_counter()
    total=np.zeros(cfg['m'],complex)
    for start_idx in range(0,len(x),1000):
        total+=np.exp(1j*(omega@x[start_idx:start_idx+1000].T)).sum(axis=1)
    z=total/(cfg['n']*np.sqrt(cfg['m']))
    sketch_seconds=time.perf_counter()-start
    side=int(np.sqrt(cfg['p']));assert side**2==cfg['p']
    grid=np.linspace(*cfg['domain'],side)
    theta0=np.array(np.meshgrid(grid,grid)).reshape(2,-1).T+rng.uniform(-.04,.04,(cfg['p'],2))
    return omega,z,theta0,sketch_seconds

def run(omega,z,theta0,method,cfg=CFG):
    theta=theta0.copy();clock=0.;solves=0;prox_steps=0;backtracks=0
    start=time.perf_counter();A=atoms(theta,omega,cfg);w=inner(A,z,cfg);solves+=1
    clock+=time.perf_counter()-start
    records=[];positions=[];weights=[]
    max_increase=0.;max_solve_kkt=kkt(A,w,z,cfg)
    def record():
        records.append([clock,value(A,w,z,cfg),ise(theta,w,cfg),np.sum(w>cfg['support_threshold']),kkt(A,w,z,cfg),np.linalg.norm(grad_theta(A,w,z,omega)),solves,prox_steps,backtracks])
        positions.append(theta.copy());weights.append(w.copy())
    record()
    for it in range(cfg['outer_steps']):
        start=time.perf_counter();old_value=value(A,w,z,cfg)
        if method=='joint':
            # Simultaneous variable-metric proximal-gradient step on the full
            # joint objective. Both candidates are computed from (theta,w).
            Q=(A.conj().T@A).real+cfg['mu']*np.eye(len(w))
            gw=Q@w-(A.conj().T@z).real
            L=np.linalg.eigvalsh(Q)[-1]
            g=grad_theta(A,w,z,omega)
            smooth=.5*np.vdot(A@w-z,A@w-z).real+.5*cfg['mu']*np.dot(w,w)
            scale=1.
            for trial in range(30):
                candidate=theta-scale*cfg['theta_step']*g
                wc=np.maximum(0.,w-scale*(gw+cfg['lam'])/L)
                Ac=atoms(candidate,omega,cfg)
                rc=Ac@wc-z
                smooth_c=.5*np.vdot(rc,rc).real+.5*cfg['mu']*np.dot(wc,wc)
                dt=candidate-theta;dw=wc-w
                model=smooth+np.sum(g*dt)+np.dot(gw,dw)+.5/scale*(np.sum(dt*dt)/cfg['theta_step']+L*np.dot(dw,dw))
                if smooth_c<=model+1e-14:break
                scale*=.5;backtracks+=1
            else:raise RuntimeError('Joint line search did not converge')
            prox_steps+=1
            theta,A,w=candidate,Ac,wc
        elif method=='palm':
            # PALM, centers first: quadratic block-majorization with safety >1.
            # Unlike prox1, the mass gradient uses the NEW centers.
            base=value(A,w,z,cfg);g=grad_theta(A,w,z,omega)
            curvature=1/cfg['theta_step'];gamma=cfg['palm_safety']
            for trial in range(30):
                candidate=theta-g/(gamma*curvature);dt=candidate-theta
                Ac=atoms(candidate,omega,cfg)
                if value(Ac,w,z,cfg)<=base+np.sum(g*dt)+.5*curvature*np.sum(dt*dt)+1e-14:break
                curvature*=2;backtracks+=1
            else:raise RuntimeError('PALM majorization failed')
            theta,A=candidate,Ac
            Q=(A.conj().T@A).real+cfg['mu']*np.eye(len(w))
            gw=Q@w-(A.conj().T@z).real
            L=np.linalg.eigvalsh(Q)[-1]
            w=np.maximum(0.,w-(gw+cfg['lam'])/(gamma*L));prox_steps+=1
        else:
            if method!='varpro':
                Q=(A.conj().T@A).real+cfg['mu']*np.eye(len(w));c=(A.conj().T@z).real-cfg['lam']
                L=np.linalg.eigvalsh(Q)[-1]
                for j in range(1 if method=='prox1' else 10):
                    w=np.maximum(0.,w-(Q@w-c)/L);prox_steps+=1
            base=value(A,w,z,cfg);g=grad_theta(A,w,z,omega)
            step=cfg['theta_step']
            for trial in range(30):
                candidate=theta-step*g;Ac=atoms(candidate,omega,cfg)
                if method=='varpro':
                    wc=inner(Ac,z,cfg);solves+=1
                else:wc=w
                vc=value(Ac,wc,z,cfg)
                if vc<=base-cfg['armijo']*step*np.sum(g*g)+1e-14:break
                step*=.5;backtracks+=1
            else:raise RuntimeError('Line search did not converge')
            theta,A,w=candidate,Ac,wc
        clock+=time.perf_counter()-start
        max_increase=max(max_increase,value(A,w,z,cfg)-old_value)
        if method=='varpro':max_solve_kkt=max(max_solve_kkt,kkt(A,w,z,cfg))
        record()
    assert max_increase<1e-11
    if method=='varpro':assert max_solve_kkt<1e-9
    return dict(history=np.array(records),theta=np.array(positions),w=np.array(weights),max_increase=max_increase,max_solve_kkt=max_solve_kkt)

def check_math(omega,z,t,cfg=CFG):
    A=atoms(t,omega,cfg);w=inner(A,z,cfg);g=grad_theta(A,w,z,omega)
    h=1e-6;fd=[]
    for j in range(t.size):
        tp=t.copy();tm=t.copy();tp.flat[j]+=h;tm.flat[j]-=h
        Ap=atoms(tp,omega,cfg);Am=atoms(tm,omega,cfg)
        fd.append((value(Ap,inner(Ap,z,cfg),z,cfg)-value(Am,inner(Am,z,cfg),z,cfg))/(2*h))
    err=float(np.max(np.abs(g.ravel()-fd)));assert err<1e-6
    assert kkt(A,w,z,cfg)<1e-9
    return {'envelope_finite_difference_max_abs_error':err,'initial_inner_kkt':kkt(A,w,z,cfg)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',default='results');parser.add_argument('--seeds',type=int,nargs='+');args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=CFG.copy()
    if args.seeds is not None:cfg['seeds']=args.seeds
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    checks=[];summary=[]
    # Warm up numerical libraries before recording timings.
    om,zz,tt,_=make_problem(0,cfg);inner(atoms(tt,om,cfg),zz,cfg)
    for j,seed in enumerate(cfg['seeds']):
        omega,z,t0,sketch_seconds=make_problem(seed,cfg)
        if j==0:checks.append(check_math(omega,z,t0,cfg))
        data={'omega':omega,'z':z,'theta0':t0,'sketch_seconds':sketch_seconds}
        # Rotate method order to reduce systematic warm-cache/order bias.
        order=METHODS[j%len(METHODS):]+METHODS[:j%len(METHODS)]
        for method in order:
            r=run(omega,z,t0,method,cfg)
            for name in ['history','theta','w']:data[method+'_'+name]=r[name]
            end=r['history'][-1]
            summary.append(dict(seed=seed,method=method,seconds=end[0],objective=end[1],ise=end[2],active=int(end[3]),kkt=end[4],outer_grad_norm=end[5],nnls_calls=int(end[6]),prox_steps=int(end[7]),backtracks=int(end[8]),mass=float(r['w'][-1].sum()),max_increase=r['max_increase'],max_solve_kkt=r['max_solve_kkt'],sketch_seconds=sketch_seconds))
        # Same-data unpenalized control, same ridge and optimizer settings.
        control=cfg.copy();control['lam']=0.
        r=run(omega,z,t0,'varpro',control)
        for name in ['history','theta','w']:data['lambda0_'+name]=r[name]
        # Classical convex fixed-dictionary reference; not the same parameter space.
        coords=np.linspace(*cfg['domain'],cfg['grid_side'])
        tg=np.array(np.meshgrid(coords,coords)).reshape(2,-1).T
        start=time.perf_counter();Ag=atoms(tg,omega,cfg);wg=inner(Ag,z,cfg)
        gridtime=time.perf_counter()-start
        data.update(grid_theta=tg,grid_w=wg,grid_seconds=gridtime)
        summary.append(dict(seed=seed,method='grid',seconds=gridtime,ise=ise(tg,wg,cfg),active=int(np.sum(wg>cfg['support_threshold'])),kkt=kkt(Ag,wg,z,cfg),mass=float(wg.sum())))
        end=r['history'][-1]
        summary.append(dict(seed=seed,method='lambda0',seconds=end[0],ise=end[2],active=int(end[3]),kkt=end[4],mass=float(r['w'][-1].sum()),max_increase=r['max_increase'],max_solve_kkt=r['max_solve_kkt']))
        np.savez_compressed(out/f'seed_{seed}.npz',**data)
        print(seed,[(s['method'],s['active'],round(s['ise'],7),round(s['seconds'],4)) for s in summary[-(len(METHODS)+2):]],flush=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    meta=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,platform=platform.platform(),processor=platform.processor(),threads=1,timing='perf_counter; optimization kernels incl. initialization, inner solves, Gram/eigensolves and line searches; excludes diagnostics and sketch construction')
    (out/'environment.json').write_text(json.dumps(meta,indent=2))

if __name__=='__main__':main()
