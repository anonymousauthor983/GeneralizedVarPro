"""Checks equations and saved results; no optimization timing is modified."""
import json
from pathlib import Path
from scipy.optimize import linear_sum_assignment
import numpy as np
from experiment import atoms,inner,kkt,grad_theta,value,ise,METHODS,check_math
R=Path(__file__).resolve().parent/'results';cfg=json.loads((R/'config.json').read_text());rows=json.loads((R/'summary.json').read_text())
report={'threshold_five_counts':{},'max_objective_increase':0.,'max_varpro_kkt':0.,'max_grid_kkt':0.}
for thr in [1e-8,1e-6,1e-4,1e-3]:
    report['threshold_five_counts'][str(thr)]={m:0 for m in METHODS+['lambda0']}
report['localization']={}
for seed in cfg['seeds']:
    d=np.load(R/f'seed_{seed}.npz');om,z=d['omega'],d['z']
    initial=[d[m+'_w'][0] for m in METHODS]
    assert all(np.array_equal(initial[0],w) for w in initial)
    for m in METHODS+['lambda0']:
        c=cfg.copy()
        if m=='lambda0':c['lam']=0
        h=d[m+'_history'];assert h.shape==(301,9)
        inc=float(np.diff(h[:,1]).max());assert inc<1e-11
        report['max_objective_increase']=max(report['max_objective_increase'],inc)
        assert np.isfinite(h).all() and (d[m+'_w']>=0).all()
        if m in ['lambda0','varpro']:
            report['max_varpro_kkt']=max(report['max_varpro_kkt'],float(h[:,4].max()));assert h[:,4].max()<1e-9
        for thr in [1e-8,1e-6,1e-4,1e-3]:report['threshold_five_counts'][str(thr)][m]+=int(np.sum(d[m+'_w'][-1]>thr)==5)
        tfin=d[m+'_theta'][-1];wfin=d[m+'_w'][-1]
        active=tfin[wfin>cfg['support_threshold']];truth=np.array(cfg['true_centers'])
        dist=np.linalg.norm(truth[:,None,:]-active[None,:,:],axis=2)
        ri,ci=linear_sum_assignment(dist)
        maximum=float(dist[ri,ci].max()) if len(ri)==5 else None
        report['localization'].setdefault(m,[]).append({'seed':seed,'active':len(active),'max_matched_distance':maximum,'compact_localized':bool(len(active)==5 and maximum is not None and maximum<cfg['sigma'])})
        # Independently regenerate final objective and analytic ISE from saved parameters.
        t,w=d[m+'_theta'][-1],d[m+'_w'][-1];A=atoms(t,om,c)
        assert abs(value(A,w,z,c)-h[-1,1])<1e-12
        assert abs(ise(t,w,c)-h[-1,2])<1e-12
    gridk=kkt(atoms(d['grid_theta'],om,cfg),d['grid_w'],z,cfg)
    assert gridk<1e-9;report['max_grid_kkt']=max(report['max_grid_kkt'],gridk)
# Check both joint partial derivative and reduced envelope derivative.
d=np.load(R/f"seed_{cfg['seeds'][0]}.npz");t,om,z=d['theta0'],d['omega'],d['z'];A=atoms(t,om,cfg)
w=np.linspace(.01,.12,len(t));g=grad_theta(A,w,z,om);fd=[];eps=1e-6
for j in range(t.size):
    tp=t.copy();tm=t.copy();tp.flat[j]+=eps;tm.flat[j]-=eps
    fd.append((value(atoms(tp,om,cfg),w,z,cfg)-value(atoms(tm,om,cfg),w,z,cfg))/(2*eps))
report['joint_gradient_max_abs_error']=float(np.max(np.abs(g.ravel()-fd)))
assert report['joint_gradient_max_abs_error']<1e-7
report.update(check_math(om,z,t,cfg))
# Independent numerical integration checks the 2D Gaussian product constant.
t,w=d['varpro_theta'][-1],d['varpro_w'][-1];w=w/w.sum();ts=np.array(cfg['true_centers']);ps=np.array(cfg['true_weights']);s=cfg['sigma']
grid=np.linspace(-3,3,601);X,Y=np.meshgrid(grid,grid)
def dens(t,w):
    out=np.zeros_like(X)
    for a,b in zip(t,w):out+=b*np.exp(-((X-a[0])**2+(Y-a[1])**2)/(2*s*s))/(2*np.pi*s*s)
    return out
numeric=float(np.trapezoid(np.trapezoid((dens(t,w)-dens(ts,ps))**2,grid,axis=1),grid))
report['ise_quadrature_abs_error']=abs(numeric-ise(t,w,cfg));assert report['ise_quadrature_abs_error']<1e-9
report['compact_localized_counts']={m:sum(x['compact_localized'] for x in rr) for m,rr in report['localization'].items()}
report['backtracks_total']={m:int(sum(r.get('backtracks',0) for r in rows if r['method']==m)) for m in METHODS}
(R/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
