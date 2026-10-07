"""Regenerate all figures and numerical tables from the saved paired runs."""
import json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from experiment import METHODS
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'reproducibility/results'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
cfg=json.loads((R/'config.json').read_text()); rows=json.loads((R/'summary.json').read_text())
data=[np.load(R/f'seed_{s}.npz') for s in cfg['seeds']]
labels={'joint':'Joint PG','prox1':'Alt-1','prox10':'Alt-10','palm':'PALM','varpro':'VarPro','grid':'Fixed grid','lambda0':r'VarPro, $\lambda=0$'}
colors=dict(zip(METHODS,['#0072B2','#E69F00','#009E73','#CC79A7','#D55E00']))
styles=dict(zip(METHODS,['-','--','-.',':','-']))
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':9,'axes.labelsize':10,'legend.fontsize':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
def finish(fig,name):
    for ext in ['pdf','png']:
        target=F/(name+'.'+ext);tmp=F/(name+'.tmp.'+ext)
        with tmp.open('wb') as out:
            fig.savefig(out,format=ext,dpi=180)
            out.flush();os.fsync(out.fileno())
        tmp.replace(target)
    plt.close(fig)
def band(ax,k,time=False,log=False,methods=METHODS):
    for m in methods:
        hs=[d[m+'_history'] for d in data]
        if time:
            # Common duration across all trials of this method; no extrapolation.
            lo=max(h[0,0] for h in hs);hi=min(h[-1,0] for h in hs)
            x=np.linspace(lo,hi,301)
            arr=np.array([np.interp(x,h[:,0],h[:,k]) for h in hs]);x=x*1000
        else:x=np.arange(301);arr=np.array([h[:,k] for h in hs])
        if log:arr=np.maximum(arr,1e-16)
        q=np.percentile(arr,[25,50,75],axis=0)
        ax.plot(x,q[1],color=colors[m],ls=styles[m],lw=1.5,label=labels[m]);ax.fill_between(x,q[0],q[2],color=colors[m],alpha=.11,lw=0)
    ax.grid(alpha=.15);ax.set_xlabel('Optimization time (ms)' if time else 'Outer updates')
    if log:ax.set_yscale('log')
def overview(methods,name):
    fig,axs=plt.subplots(1,3,figsize=(7.05,2.05),layout='constrained')
    band(axs[0],3,methods=methods);axs[0].set_ylabel('Active components')
    ymax=max(np.percentile([d[m+'_history'][0,3] for d in data],75) for m in methods)
    axs[0].set_ylim(4.5,ymax+1);axs[0].set_yticks(np.arange(5,ymax+1,5))
    band(axs[1],2,log=True,methods=methods);axs[1].set_ylabel('Density ISE')
    times=[np.array([r['seconds']*1000 for r in rows if r['method']==m]) for m in methods]
    bp=axs[2].boxplot(times,positions=np.arange(len(methods)),widths=.5,showfliers=False,patch_artist=True)
    for j,(m,v) in enumerate(zip(methods,times)):
        bp['boxes'][j].set(facecolor=colors[m],alpha=.3)
        axs[2].scatter(j+np.linspace(-.13,.13,len(v)),v,s=5,color=colors[m],alpha=.7)
    axs[2].set_xticks(range(len(methods)),[labels[m] for m in methods],rotation=30,ha='right')
    axs[2].set_ylabel('Time for 300 updates (ms)');axs[2].grid(axis='y',alpha=.15)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='outside upper center',ncol=len(methods),frameon=False)
    finish(fig,name)
overview(['joint','palm','varpro'],'fig_sketch_varpro')
overview(METHODS,'fig_sketch_all_methods')
fig,axs=plt.subplots(2,2,figsize=(7.05,4.2),layout='constrained')
band(axs[0,0],4,log=True);axs[0,0].set_ylabel('Mass KKT residual')
for m in METHODS:
    arr=[]
    for d in data:
        best=min(d[n+'_history'][:,1].min() for n in METHODS)
        arr.append(np.maximum(d[m+'_history'][:,1]-best,1e-15))
    q=np.percentile(arr,[25,50,75],axis=0);axs[0,1].plot(q[1],color=colors[m],ls=styles[m]);axs[0,1].fill_between(np.arange(301),q[0],q[2],color=colors[m],alpha=.11)
axs[0,1].set_yscale('log');axs[0,1].set_ylabel('Objective minus best observed');axs[0,1].set_xlabel('Outer updates');axs[0,1].grid(alpha=.15)
band(axs[1,0],2,time=True,log=True);axs[1,0].set_ylabel('Density ISE')
band(axs[1,1],3,time=True);axs[1,1].set_ylabel('Active components')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=5,frameon=False)
finish(fig,'fig_sketch_diagnostics')
fig,axs=plt.subplots(1,3,figsize=(7.05,2.05),layout='constrained')
for ax,k,yl in zip(axs,[3,2,0],['Active components','Density ISE','Optimization time (ms)']):
    a=np.array([d['lambda0_history'][-1,k] for d in data]);b=np.array([d['varpro_history'][-1,k] for d in data])
    if k==0:a*=1000;b*=1000
    for aa,bb in zip(a,b):ax.plot([0,1],[aa,bb],color='.75',lw=.65,zorder=1)
    ax.boxplot([a,b],positions=[0,1],widths=.45,showfliers=False)
    ax.scatter(np.zeros(20),a,s=6,color='#0072B2');ax.scatter(np.ones(20),b,s=6,color=colors['varpro'])
    ax.set_xticks([0,1],[r'$\lambda=0$',r'$\lambda=0.008$']);ax.set_ylabel(yl);ax.grid(axis='y',alpha=.15)
finish(fig,'fig_sketch_lambda_control')
# Spatial contours are generated separately by export_contours.py and TikZ.
# A machine-readable summary is the sole source of numbers used in the text.
agg={}
for m in METHODS+['grid','lambda0']:
    rr=[r for r in rows if r['method']==m]
    agg[m]={k:np.percentile([r[k] for r in rr],[25,50,75]).tolist() for k in ['ise','active','seconds','mass','kkt']}
    agg[m]['five']=sum(r['active']==5 for r in rr)
agg['sketch_seconds']=np.percentile([float(d['sketch_seconds']) for d in data],[25,50,75]).tolist()
(R/'aggregate.json').write_text(json.dumps(agg,indent=2))
print(json.dumps(agg,indent=2))

# Regenerate the appendix table, including fresh timings after rerunning.
s=r"""\begin{table}[ht]
\centering\small
\caption{Final results over 20 paired trials. Brackets give interquartile
ranges. Learned dictionaries use 300 mean updates; the fixed grid uses one
convex solve. All ISE values use normalized masses.}
\label{tab:sketch-results}
\begin{tabular}{lrrrr}
\toprule
Method & Five atoms & Active atoms & ISE ($\times10^{-4}$) & Time (ms)\\
\midrule
"""
for m in METHODS+['grid','lambda0']:
    def fmt(k,scale,digits):
        q=[v*scale for v in agg[m][k]]
        if k=='active':return f'{q[1]:g}\\,[{q[0]:g},{q[2]:g}]'
        return f'{q[1]:.{digits}f}\\,[{q[0]:.{digits}f},{q[2]:.{digits}f}]'
    s+=f'{labels[m]} & ${agg[m]["five"]}/20$ & ${fmt("active",1,2)}$ & ${fmt("ise",1e4,2)}$ & ${fmt("seconds",1e3,1)}$'+r'\\'+'\n'
s+=r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table}'+'\n'
(ROOT/'sections/sketching_results_table.tex').write_text(s)
