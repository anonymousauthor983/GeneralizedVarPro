"""Reconstruct the paper's validation slice 24 from reproduced checkpoints."""
import argparse,json
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import FeatureNet,acquisition,features_input,gram,solve_codes,synth
from data import load
import metrics_ops as tv
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--runs',default='runs');p.add_argument('--output',default='figures');p.add_argument('--device',default='cuda');a=p.parse_args()
torch.set_num_threads(4);arrays,_=load({'data':a.data});x=arrays['val'][24:25].to(a.device);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,axs=plt.subplots(4,6,figsize=(12,9));records=[]
labels=['Reference','Zero-filled','VarPro','Joint prox.','Alternating','TV']
for row,R in enumerate([4,8]):
 mask=tv.cartesian_mask(320,R,100,a.device);y=acquisition(x,mask)
 images=[x,torch.fft.ifft2(y,norm='ortho').real]
 with torch.no_grad():
  for method in ['projected','joint_prox','alternating']:
   path=Path(a.runs)/'seed0'/'mu1'/'results'/f'r{R}_p1_{method}_l1'/'latest.pt'
   cp=torch.load(path,map_location=a.device,weights_only=False)
   net=FeatureNet().to(a.device);net.load_state_dict(cp['model']);net.eval()
   B=net(features_input(y));H,b=gram(B,y,mask);w,_=solve_codes(H,b,cp['lambda_train'],power=1.)
   images.append(synth(B,w))
  rec,_,_=tv.solve(y,mask,torch.ones_like(tv.differences(x)),.04,tol=1e-5,max_iter=240000);images.append(rec)
 for j,im in enumerate(images):
  ax=axs[2*row,j];ax.imshow(im[0,0].abs().cpu(),cmap='gray',vmin=0,vmax=4.5);ax.axis('off')
  if row==0:ax.set_title(labels[j])
  err=axs[2*row+1,j];err.axis('off')
  if j==0:err.text(.5,.5,f'×{R}\nAbsolute error',ha='center',va='center',transform=err.transAxes)
  else:
   psnr=float(tv.image_metrics(im,x)['psnr'][0]);ax.text(.5,-.04,f'{psnr:.2f} dB',ha='center',va='top',transform=ax.transAxes)
   heat=err.imshow((im.abs()-x).abs()[0,0].cpu(),cmap='inferno',vmin=0,vmax=.5)
   records.append(dict(R=R,method=labels[j],psnr=psnr))
fig.subplots_adjust(left=.02,right=.98,top=.94,bottom=.1,hspace=.22,wspace=.04)
cax=fig.add_axes([.35,.035,.4,.015]);fig.colorbar(heat,cax=cax,orientation='horizontal',label='Absolute error')
fig.savefig(out/'mri_reconstructions_slice24.pdf',dpi=300);fig.savefig(out/'mri_reconstructions_slice24.png',dpi=180)
(out/'slice24_metrics.json').write_text(json.dumps(records,indent=2))
