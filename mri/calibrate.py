import argparse,json
from pathlib import Path
import torch
import numpy as np
from core import *
from data import load
from metrics_ops import cartesian_mask
from check_powers import check

def main(family):
 c=json.loads(Path('config.json').read_text());R=c['accelerations'][family//3];power=c['powers'][family%3]
 tests=check(power,'cuda');torch.set_num_threads(4);torch.manual_seed(0);a,selection=load(c)
 if family==0:Path('selection.json').write_text(json.dumps(selection,indent=2))
 net=FeatureNet().cuda();mask=cartesian_mask(320,R,0,'cuda');scales=[]
 with torch.no_grad():
  for i in range(0,32,4):
   x=a['train'][i:i+4].cuda();y=acquisition(x,mask);B=net(features_input(y));H,b=gram(B,y,mask)
   if power==1:v=b.abs().amax(1)
   elif power==.5:v=((2*b.abs()/3).pow(3)/H.diagonal(dim1=-2,dim2=-1)).sqrt().amax(1)
   else:v=torch.linalg.eigvalsh(H).amax(1)/2
   scales.extend(v.tolist())
 scale=float(np.median(scales));fractions=[.01,.1,1.] if power==2 else [.001,.01,.1];lambdas=[scale*f for f in fractions];checks=[]
 for lam in lambdas:
  torch.manual_seed(0);net=FeatureNet().cuda();opt=torch.optim.Adam(net.parameters(),lr=c['lr'])
  for step in range(3):
   x=a['train'][4*step:4*step+4].cuda();y=acquisition(x,mask);B=net(features_input(y));H,b=gram(B,y,mask)
   w,st=solve_codes(H,b,lam,power=power);wi=implicit_codes(H,b,w,lam,st,power=power)
   opt.zero_grad();loss=(synth(B,wi)-x).square().mean()/2;loss.backward()
   assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters());opt.step()
   row=dict(R=R,power=power,lam=lam,step=step,loss=float(loss),states=st);checks.append(row);print(json.dumps(row),flush=True)
  x=a['train'][:1].cuda();y=acquisition(x,mask);net.zero_grad();B=net(features_input(y));H,b=gram(B,y,mask)
  w,st=solve_codes(H,b,lam,power=power);wi=implicit_codes(H,b,w,lam,st,power=power)
  loss=(synth(B,wi)-x).square().mean()/2;loss.backward();analytic=float(net.output.bias.grad[0]);vals=[]
  with torch.no_grad():
   original=net.output.bias.clone()
   for sign in [-1,1]:
    net.output.bias.copy_(original);net.output.bias[0]+=sign*.001;B=net(features_input(y));H,b=gram(B,y,mask)
    # Half: warm-start selected local branch for local sensitivity, not global claims.
    wp,_=solve_codes(H,b,lam,initial=w if power==.5 else None,power=power)
    vals.append(float((synth(B,wp)-x).square().mean()/2))
   net.output.bias.copy_(original)
  fd=(vals[1]-vals[0])/.002;assert abs(fd-analytic)<max(2e-3,.03*abs(fd)),(power,lam,fd,analytic)
  checks.append(dict(lam=lam,analytic=analytic,finite_difference=fd))
 Path(f'CALIBRATION_{family}.json').write_text(json.dumps(dict(passed=True,R=R,power=power,scale=scale,fractions=fractions,lambdas=lambdas,lambda_test_grid=lambdas,checks=checks,unit_tests=tests),indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--family',type=int,required=True);main(ap.parse_args().family)
