"""Exploratory design search ONLY on seeds 0--4; all results are logged.
Evaluation seeds 201--220 are reserved, not accessed by this script.
"""
from pathlib import Path
import json,itertools
from experiment import make_problem,run,METHODS
CFG={'n': 20000, 'm': 128, 'p': 16, 'sigma': 0.15, 'mu': 0.0001, 'lam': 0.008, 'outer_steps': 300, 'theta_step': 0.5, 'armijo': 0.0001, 'support_threshold': 1e-06, 'seeds': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120], 'dimension': 2, 'frequency_scale': 3.0, 'palm_safety': 1.1, 'grid_side': 21, 'domain': [-1.5, 1.5], 'true_centers': [[-1.0, -0.8], [-1.0, 0.8], [0.1, 0.0], [1.0, -0.8], [1.0, 0.8]], 'true_weights': [0.28, 0.24, 0.2, 0.16, 0.12]}
OUT=Path(__file__).resolve().parent/'exploration';OUT.mkdir(exist_ok=True)
# Two candidate overparameterizations, three common penalties; all optimizers
# retain the prior bundle's step sizes and 300-update budget.
base=CFG.copy();base.update(true_centers=[[-1.,-.8],[-1.,.8],[.1,0.],[1.,-.8],[1.,.8]],true_weights=[.28,.24,.2,.16,.12],seeds=list(range(5)))
configs=[];rows=[]
for p,lam in itertools.product([16,25],[.004,.008,.012]):
 c=base.copy();c.update(p=p,lam=lam);configs.append(c)
 for seed in c['seeds']:
  om,z,t,_=make_problem(seed,c)
  for m in ['joint','palm','varpro']:
   r=run(om,z,t,m,c);h=r['history'][-1]
   rows.append(dict(config=len(configs)-1,seed=seed,method=m,active=int(h[3]),ise=float(h[2]),objective=float(h[1]),max_kkt=r['max_solve_kkt']))
 (OUT/'candidate_configs.json').write_text(json.dumps(configs,indent=2));(OUT/'candidate_results.json').write_text(json.dumps(rows,indent=2))
 print(len(configs)-1,p,lam,{m:[r['active'] for r in rows if r['config']==len(configs)-1 and r['method']==m] for m in ['joint','palm','varpro']},flush=True)
