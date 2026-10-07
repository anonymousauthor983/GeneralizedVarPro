import argparse,json,time
from pathlib import Path
import torch
from data import load
from core import acquisition
import metrics_ops as tv
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',default='tv_results.json');p.add_argument('--device',default='cuda');a=p.parse_args()
torch.set_num_threads(4);arrays,_=load({'data':a.data});rows=[]
for R in [4,8]:
 mask=tv.cartesian_mask(320,R,100,a.device)
 for lam in [.01,.04,.16]:
  metrics=[];states=[];start=time.perf_counter()
  with torch.no_grad():
   for i in range(0,480,4):
    x=arrays['val'][i:i+4].to(a.device);y=acquisition(x,mask)
    rec,_,st=tv.solve(y,mask,torch.ones_like(tv.differences(x)),lam,tol=1e-5,max_iter=240000)
    mm=tv.image_metrics(rec,x);metrics.extend([{k:float(v[j]) for k,v in mm.items()} for j in range(len(x))]);states.append(st)
  rows.append(dict(R=R,lambda_tv=lam,validation={k:sum(r[k] for r in metrics)/480 for k in metrics[0]},per_image=metrics,solver_states=states,seconds=time.perf_counter()-start))
  Path(a.output).write_text(json.dumps(rows,indent=2))
