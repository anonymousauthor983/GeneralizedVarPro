import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs');p.add_argument('--output',default='summary.json');a=p.parse_args();groups={}
for f in Path(a.runs).glob('seed*/mu*/results/*/result.json'):
 d=json.loads(f.read_text());ev=[e for e in d['all_inference_results'] if e['lambda_test']==d['lambda_train']];assert len(ev)==1
 for mu in ([0,1] if d['method']=='projected' and d['power']==1 else [d['mu']]):
  key=(d['R'],d['power'],d['lambda_train'],mu);groups.setdefault(key,{}).setdefault(d['method'],{})[d['seed']]=ev[0]['validation']
rows=[]
for key,methods in sorted(groups.items()):
 names=['projected','joint_prox','alternating'];seeds=sorted(set.intersection(*(set(methods.get(name,{})) for name in names)))
 for name in names:
  vals=[methods[name][s] for s in seeds];metrics={}
  if vals:
   for k in vals[0]:
    v=[r[k] for r in vals];metrics[k]={'mean':statistics.mean(v),'variance':statistics.variance(v) if len(v)>1 else None,'std':statistics.stdev(v) if len(v)>1 else None}
  rows.append(dict(R=key[0],power=key[1],lambda_value=key[2],mu=key[3],method=name,seeds=seeds,metrics=metrics))
Path(a.output).write_text(json.dumps(rows,indent=2))
