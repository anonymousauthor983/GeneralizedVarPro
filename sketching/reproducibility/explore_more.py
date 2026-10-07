"""Second exploratory round, still only seeds 0--4. No evaluation data."""
from pathlib import Path
import json,itertools
from experiment import make_problem,run
CFG={'n': 20000, 'm': 128, 'p': 16, 'sigma': 0.15, 'mu': 0.0001, 'lam': 0.008, 'outer_steps': 300, 'theta_step': 0.5, 'armijo': 0.0001, 'support_threshold': 1e-06, 'seeds': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120], 'dimension': 2, 'frequency_scale': 3.0, 'palm_safety': 1.1, 'grid_side': 21, 'domain': [-1.5, 1.5], 'true_centers': [[-1.0, -0.8], [-1.0, 0.8], [0.1, 0.0], [1.0, -0.8], [1.0, 0.8]], 'true_weights': [0.28, 0.24, 0.2, 0.16, 0.12]}
OUT=Path(__file__).resolve().parent/'exploration'
configs=json.loads((OUT/'candidate_configs.json').read_text());rows=json.loads((OUT/'candidate_results.json').read_text())
geometries=[([[-1.,-.6],[.2,.8],[1.1,-.5],[-1.,.9],[.1,-1.]], [.40,.18,.16,.14,.12]),([[-1.,-.8],[-1.,.8],[.1,0.],[1.,-.8],[1.,.8]],[.44,.18,.15,.13,.10])]
for (centers,weights),p in itertools.product(geometries,[16,36]):
 c=CFG.copy();c.update(true_centers=centers,true_weights=weights,p=p,seeds=list(range(5)))
 configs.append(c);index=len(configs)-1
 for seed in c['seeds']:
  om,z,t,_=make_problem(seed,c)
  for m in ['joint','palm','varpro']:
   r=run(om,z,t,m,c);h=r['history'][-1]
   rows.append(dict(config=index,seed=seed,method=m,active=int(h[3]),ise=float(h[2]),objective=float(h[1]),max_kkt=r['max_solve_kkt']))
 (OUT/'candidate_configs.json').write_text(json.dumps(configs,indent=2));(OUT/'candidate_results.json').write_text(json.dumps(rows,indent=2))
 print(index,p,{m:[r['active'] for r in rows if r['config']==index and r['method']==m] for m in ['joint','palm','varpro']},flush=True)
