"""Image-specific normalized synthesis features, exact active-set differentiation.
No imposed spatial derivative, blocks, orthogonality, or added ridge.
"""
import torch
import numpy as np
from torch import nn
from unet import UNet

class FeatureNet(UNet):
    def __init__(self):
        super().__init__();self.output=nn.Conv2d(32,16,1)
    def forward(self,u):
        raw=super().forward(u).double()
        norm=raw.square().sum((-2,-1),keepdim=True).sqrt()
        if float(norm.detach().min())<1e-10:raise RuntimeError('Null feature column; normalization undefined')
        return raw/norm

def features_input(y):
    z=torch.fft.ifft2(y,norm='ortho');return torch.cat((z.real,z.imag),1).float()

def acquisition(x,mask):return torch.fft.fft2(x,norm='ortho')*mask

def synth(B,w):return torch.einsum('bdhw,bd->bhw',B,w)[:,None]

def gram(B,y,mask,x=None,mu=1.):
    # Return Hessian and RHS of .5 w'Hw - b'w + lambda ||w||1.
    Z=acquisition(B,mask).flatten(2);Y=y.flatten(2)
    H=(Z.conj()@Z.transpose(1,2)).real
    b=(Z.conj()@Y.transpose(1,2)).real.squeeze(-1)
    if x is not None:
        P=B.flatten(2);H=P@P.transpose(1,2)+mu*H
        b=(P@x.flatten(2).transpose(1,2)).squeeze(-1)+mu*b
    return H,b

def diagnostics(H,b,w,lam):
    g=H@w-b;active=w!=0
    r=np.where(active,np.abs(g+lam*np.sign(w)),np.maximum(np.abs(g)-lam,0))
    scale=max(float(np.max(np.abs(b))),1.)
    resid=float(np.max(r))/scale
    if np.any(active):
        eig=np.linalg.eigvalsh(H[np.ix_(active,active)])
        ratio=float(eig[0]/max(eig[-1],1e-300))
    else:ratio=1.
    margin=float(np.min(lam-np.abs(g[~active]))/scale) if np.any(~active) else float('inf')
    return dict(kkt=resid,active=int(active.sum()),active_eigen_ratio=ratio,inactive_margin=margin)

def lasso_active_refine(h,b,w,lam,tol,max_steps=2000):
    """Bound-constrained active-set solve of the unchanged Lasso.

    w = u-v, u,v >= 0. Solve on the free set to positivity BEFORE admitting
    another violating variable. Unlike a coordinate sweep between corrections,
    this does not repeatedly reintroduce a variable just removed at a boundary.
    No diagonal shift, smoothing or relaxation of the final Lasso KKT check.
    """
    n=len(w);Q=np.block([[h,-h],[-h,h]])
    c=np.concatenate((b-lam,-b-lam))
    x=np.concatenate((np.maximum(w,0),np.maximum(-w,0)))
    free=x>0
    for iteration in range(max_steps):
        # Fully optimize the current face, dropping negative free variables.
        for inner in range(4*n+1):
            ids=np.flatnonzero(free)
            if not len(ids):break
            hs=Q[np.ix_(ids,ids)]
            try:z=np.linalg.solve(hs,c[ids])
            except np.linalg.LinAlgError:z=np.linalg.lstsq(hs,c[ids],rcond=None)[0]
            if np.all(z>0):
                x[:]=0.;x[ids]=z;break
            bad=z<=0
            old=x[ids];denom=old[bad]-z[bad]
            ratios=np.divide(old[bad],denom,out=np.zeros_like(denom),where=denom>0)
            alpha=float(np.clip(np.min(ratios),0,1))
            x[ids]=old+alpha*(z-old)
            hit=ids[np.flatnonzero(bad)[np.argmin(ratios)]]
            x[hit]=0.;free[hit]=False
        else:raise RuntimeError('Lasso face solve exceeded finite bound')
        w[:]=x[:n]-x[n:]
        st=diagnostics(h,b,w,lam)
        if st['kkt']<=tol:return st,iteration+1
        reduced=Q@x-c;reduced[free]=np.inf
        enter=int(np.argmin(reduced))
        if not np.isfinite(reduced[enter]) or reduced[enter]>=0:
            raise RuntimeError('Lasso face residual unresolved: '+str(st))
        free[enter]=True
    raise RuntimeError('Lasso bound-active solve did not converge: '+str(st))

def solve_codes(H,b,lam,initial=None,tol=1e-8,max_sweeps=20000):
    hh=H.detach().cpu().numpy();bb=b.detach().cpu().numpy()
    ww=np.zeros_like(bb) if initial is None else initial.detach().cpu().numpy().copy()
    statuses=[]
    for i,(h,bi) in enumerate(zip(hh,bb)):
        h=.5*(h+h.T);w=ww[i]
        if np.any(np.diag(h)<=1e-14):raise RuntimeError('Unobservable/null column in code solve')
        for sweep in range(max_sweeps):
            for j in range(len(w)):
                t=bi[j]-h[j]@w+h[j,j]*w[j]
                w[j]=np.sign(t)*max(abs(t)-lam,0)/h[j,j]
            if (sweep+1)%10==0:
                active=w!=0
                if np.any(active):
                    hs=h[np.ix_(active,active)];ev=np.linalg.eigvalsh(hs)
                    if ev[0]>1e-12*ev[-1]:
                        candidate=np.linalg.solve(hs,bi[active]-lam*np.sign(w[active]))
                        if np.all(candidate*w[active]>0):w[active]=candidate
                st=diagnostics(h,bi,w,lam)
                if st['kkt']<=tol:break
        else:
            before=w.copy()
            st,steps=lasso_active_refine(h,bi,w,lam,tol)
            st.update(fallback='bound_active_qp',fallback_steps=steps)
            # Preserve actual hard systems for independent regression checks.
            import os,uuid
            from pathlib import Path
            audit=os.environ.get('MRI_SOLVER_AUDIT_DIR',
                                 'solver_audit' if os.environ.get('SLURM_JOB_ID') else None)
            if audit:
                dest=Path(audit);dest.mkdir(parents=True,exist_ok=True)
                np.savez(dest/('lasso_'+uuid.uuid4().hex+'.npz'),H=h,b=bi,lam=lam,
                         before=before,solution=w.copy(),tol=tol)
        st.update(sweeps=sweep+1);statuses.append(st)
    return torch.tensor(ww,device=H.device,dtype=H.dtype),statuses

def implicit_codes(H,b,w,lam,statuses):
    """Local derivative with stable active set; stop if derivative is ill-defined."""
    answers=[]
    for i,st in enumerate(statuses):
        if st['active_eigen_ratio']<1e-10:raise RuntimeError('Singular/ill-conditioned active Hessian; no ridge silently added')
        if st['inactive_margin']<1e-9:
            # A 1e-8 KKT tolerance can leave a weakly violating inactive variable.
            # Refine the SAME Lasso before applying the unchanged 1e-9 margin guard.
            h=H[i].detach().cpu().numpy();h=.5*(h+h.T)
            bi=b[i].detach().cpu().numpy();wi=w[i].detach().cpu().numpy().copy();before=wi.copy()
            refined,steps=lasso_active_refine(h,bi,wi,lam,1e-13)
            import os,uuid
            from pathlib import Path
            dest=Path(os.environ.get('MRI_SOLVER_AUDIT_DIR','solver_audit'));dest.mkdir(parents=True,exist_ok=True)
            np.savez(dest/('margin_'+uuid.uuid4().hex+'.npz'),H=h,b=bi,lam=lam,before=before,solution=wi,tol=1e-13)
            refined.update(fallback='tight_lasso_before_derivative',fallback_steps=steps,pre_refinement_margin=st['inactive_margin'])
            w[i].copy_(torch.tensor(wi,device=w.device,dtype=w.dtype));statuses[i].update(refined);st=statuses[i]
            if st['active_eigen_ratio']<1e-10:raise RuntimeError('Refined active Hessian ill-conditioned; no ridge added')
            if st['inactive_margin']<1e-9:raise RuntimeError('Support transition after tight Lasso solve; classical implicit derivative not certified: '+str(st))
        ids=torch.nonzero(w[i]!=0,as_tuple=False).flatten()
        if len(ids):
            h=H[i].index_select(0,ids).index_select(1,ids)
            z=torch.linalg.solve(h,b[i,ids]-lam*w[i,ids].sign())
            rec=torch.zeros_like(w[i]).index_copy(0,ids,z)
        else:rec=b[i]*0
        answers.append(w[i].detach()+rec-rec.detach())
    return torch.stack(answers)

def prox_step(H,b,w,lam):
    step=.9/torch.linalg.eigvalsh(H).amax(-1).clamp_min(1e-12)
    u=w-step[:,None]*(torch.einsum('bij,bj->bi',H,w)-b)
    return u.sign()*(u.abs()-step[:,None]*lam).clamp_min(0)

# Generalization to p=2 and p=1/2, preserving the validated p=1 routines.
_lasso_solve=solve_codes
_lasso_implicit=implicit_codes
_lasso_prox=prox_step

def half_coordinate(r,h,lam):
    threshold=1.5*(h*lam*lam)**(1./3.)
    if abs(r)<=threshold:return 0.
    a=abs(r)/(3*h)
    arg=np.clip(-lam/(4*h*a**1.5),-1.,1.)
    t=2*np.sqrt(a)*np.cos(np.arccos(arg)/3.)
    return np.sign(r)*t*t

def half_diagnostics(h,b,w,lam):
    g=h@w-b;active=w!=0;scale=max(float(np.max(np.abs(b))),1.)
    grad=g[active]+.5*lam*np.sign(w[active])/np.sqrt(np.abs(w[active]))
    residual=float(np.max(np.abs(grad)))/scale if np.any(active) else 0.
    fixed=np.array([half_coordinate(b[j]-h[j]@w+h[j,j]*w[j],h[j,j],lam) for j in range(len(w))])
    coord=float(np.max(np.abs(fixed-w)))/max(float(np.max(np.abs(w))),1.)
    if np.any(active):
        hs=h[np.ix_(active,active)]-.25*lam*np.diag(np.abs(w[active])**(-1.5));ev=np.linalg.eigvalsh(hs)
        ratio=float(ev[0]/max(ev[-1],1e-300))
    else:ratio=1.
    inactive=~active
    thresholds=1.5*(np.diag(h)*lam*lam)**(1./3.)
    margin=float(np.min(thresholds[inactive]-np.abs(g[inactive]))/scale) if np.any(inactive) else float('inf')
    return dict(active_stationarity=residual,coordinate_residual=coord,active=int(active.sum()),active_eigen_ratio=ratio,inactive_margin=margin,global_optimum_certified=False)

def half_active_refine(h,b,w,lam,tol,max_steps=4000):
    """Local rescue, including descent out of negative-curvature active faces.

    All candidates are compared in the original nonsmoothed objective. Boundary
    steps set the first crossing coefficient exactly to zero. No global claim.
    """
    def change(old,new):
        delta=new-old
        return float((h@old-b)@delta+.5*delta@h@delta+
                     lam*np.sum(np.sqrt(np.abs(new))-np.sqrt(np.abs(old))))
    for iteration in range(max_steps):
        for j in range(len(w)):
            r=b[j]-h[j]@w+h[j,j]*w[j]
            w[j]=half_coordinate(r,h[j,j],lam)
        ids=np.flatnonzero(w!=0)
        if len(ids):
            old=w.copy();wa=w[ids]
            K=h[np.ix_(ids,ids)]-.25*lam*np.diag(np.abs(wa)**(-1.5))
            eig,vec=np.linalg.eigh(K)
            grad=(h@w-b)[ids]+.5*lam*np.sign(wa)/np.sqrt(np.abs(wa))
            directions=[]
            if eig[0]>1e-12*max(eig[-1],1e-300):
                directions.append(-np.linalg.solve(K,grad))
            else:
                amplitude=max(float(np.linalg.norm(wa)),1.)
                directions.extend([amplitude*vec[:,0],-amplitude*vec[:,0]])
                directions.append(-grad/max(float(np.linalg.norm(grad)),1e-300)*amplitude)
            best=0.;chosen=None
            for direction in directions:
                crossing=wa*direction<0
                boundary=float(np.min(-wa[crossing]/direction[crossing])) if np.any(crossing) else float('inf')
                limit=min(1.,boundary)
                for k in range(32):
                    step=limit*(.5**k);candidate=old.copy();candidate[ids]=wa+step*direction
                    if k==0 and boundary<=1:
                        hit=np.flatnonzero(crossing)[np.argmin(-wa[crossing]/direction[crossing])]
                        candidate[ids[hit]]=0.
                    delta=change(old,candidate)
                    if delta<best:best=delta;chosen=candidate
            if chosen is not None:w[:]=chosen
        st=half_diagnostics(h,b,w,lam)
        if max(st['active_stationarity'],st['coordinate_residual'])<=tol and st['active_eigen_ratio']>=0:
            return st,iteration+1
    raise RuntimeError('Half local curvature rescue did not converge: '+str(st))

def solve_codes(H,b,lam,initial=None,tol=1e-8,max_sweeps=20000,power=1.):
    if power==1:return _lasso_solve(H,b,lam,initial,tol,max_sweeps)
    if power==2:
        hs=H.detach()+2*lam*torch.eye(H.shape[-1],device=H.device,dtype=H.dtype)
        w=torch.linalg.solve(hs,b.detach().unsqueeze(-1)).squeeze(-1)
        ev=torch.linalg.eigvalsh(hs);res=(torch.einsum('bij,bj->bi',hs,w)-b.detach()).abs().amax(-1)/b.detach().abs().amax(-1).clamp_min(1.)
        assert float(res.max())<=tol
        st=[dict(kkt=float(res[i]),active=int((w[i]!=0).sum()),active_eigen_ratio=float(ev[i,0]/ev[i,-1]),inactive_margin=float('inf'),sweeps=1) for i in range(len(w))]
        return w,st
    assert power==.5
    hh=H.detach().cpu().numpy();bb=b.detach().cpu().numpy()
    ww=np.zeros_like(bb) if initial is None else initial.detach().cpu().numpy().copy();states=[]
    for i,(h,bi) in enumerate(zip(hh,bb)):
        h=.5*(h+h.T);w=ww[i]
        if np.any(np.diag(h)<=1e-14):raise RuntimeError('Null/unobservable column')
        value=lambda v:.5*v@h@v-bi@v+lam*np.sqrt(np.abs(v)).sum()
        for sweep in range(max_sweeps):
            for j in range(len(w)):
                r=bi[j]-h[j]@w+h[j,j]*w[j];w[j]=half_coordinate(r,h[j,j],lam)
            if (sweep+1)%10==0:
                # Newton polish on a strictly positive-curvature active branch.
                act=w!=0
                if np.any(act):
                    for _ in range(5):
                        wa=w[act];hs=h[np.ix_(act,act)]-.25*lam*np.diag(np.abs(wa)**(-1.5));ev=np.linalg.eigvalsh(hs)
                        if ev[0]<=1e-12*ev[-1]:break
                        grad=(h@w-bi)[act]+.5*lam*np.sign(wa)/np.sqrt(np.abs(wa))
                        delta=np.linalg.solve(hs,grad);old=value(w);accepted=False
                        for factor in [1.,.5,.25,.125,.0625]:
                            candidate=w.copy();candidate[act]=wa-factor*delta
                            if np.all(candidate[act]*wa>0) and value(candidate)<=old+1e-12*max(1.,abs(old)):
                                w[:]=candidate;accepted=True;break
                        if not accepted:break
                st=half_diagnostics(h,bi,w,lam)
                if max(st['active_stationarity'],st['coordinate_residual'])<=tol:
                    if st['active_eigen_ratio']<0:
                        st,steps=half_active_refine(h,bi,w,lam,tol)
                        st.update(fallback='stationary_saddle_curvature_descent',fallback_steps=steps)
                    break
        else:
            before=w.copy()
            import os,uuid
            from pathlib import Path
            dest=Path(os.environ.get('MRI_SOLVER_AUDIT_DIR','solver_audit'))
            dest.mkdir(parents=True,exist_ok=True)
            case=dest/('half_'+uuid.uuid4().hex+'.npz')
            np.savez(case,H=h,b=bi,lam=lam,before=before,tol=tol)
            st,steps=half_active_refine(h,bi,w,lam,tol)
            st.update(fallback='local_curvature_descent',fallback_steps=steps)
            np.savez(case,H=h,b=bi,lam=lam,before=before,solution=w.copy(),tol=tol)
        st.update(sweeps=sweep+1);states.append(st)
    return torch.tensor(ww,device=H.device,dtype=H.dtype),states

def implicit_codes(H,b,w,lam,statuses,power=1.):
    if power==1:return _lasso_implicit(H,b,w,lam,statuses)
    if power==2:
        return torch.linalg.solve(H+2*lam*torch.eye(H.shape[-1],device=H.device,dtype=H.dtype),b.unsqueeze(-1)).squeeze(-1)
    assert power==.5;out=[]
    for i,st in enumerate(statuses):
        if st['active_eigen_ratio']<1e-10 or st['inactive_margin']<1e-9:raise RuntimeError('Half implicit derivative not certified on selected local branch: '+str(st))
        ids=torch.nonzero(w[i]!=0,as_tuple=False).flatten()
        if len(ids):
            h=H[i].index_select(0,ids).index_select(1,ids);wa=w[i,ids].detach()
            K=h.detach()-.25*lam*torch.diag(wa.abs().pow(-1.5))
            residual=h@wa-b[i,ids]+.5*lam*wa.sign()/wa.abs().sqrt()
            delta=-torch.linalg.solve(K,residual)
            rec=torch.zeros_like(w[i]).index_copy(0,ids,delta)
        else:rec=b[i]*0
        out.append(w[i].detach()+rec-rec.detach())
    return torch.stack(out)

def prox_step(H,b,w,lam,power=1.):
    if power==1:return _lasso_prox(H,b,w,lam)
    step=.9/torch.linalg.eigvalsh(H).amax(-1).clamp_min(1e-12)
    u=w-step[:,None]*(torch.einsum('bij,bj->bi',H,w)-b)
    if power==2:return u/(1+2*step[:,None]*lam)
    threshold=1.5*(step[:,None]*lam).pow(2./3.)
    a=u.abs().clamp_min(1e-100)/3.
    arg=(-step[:,None]*lam/(4*a.pow(1.5))).clamp(-1,1)
    t=2*a.sqrt()*torch.cos(torch.acos(arg)/3.)
    return torch.where(u.abs()>threshold,u.sign()*t.square(),torch.zeros_like(u))
