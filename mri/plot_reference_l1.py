"""Publication layout; preserve the exact frozen numerical data of the prior figure."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
ROOT=Path(__file__).resolve().parent
(ROOT/'figures').mkdir(exist_ok=True)
source=ROOT/'reference/mri_l1_results_data.json'
d=json.loads(source.read_text());points=d['points'];assert len(points)==30
# Lambda values are fixed across seeds and unchanged from the original protocol.
lambdas={4:[.3649707354557575,3.6497073545575747,36.497073545575745],8:[.34366741379346366,3.4366741379346366,34.366741379346365]}
colors={'projected':'#2468A0','joint_prox':'#C87528','alternating':'#21866D'}
labels={'projected':'VarPro','joint_prox':'Joint proximal','alternating':'Alternating'}
markers=['o','s','^']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'axes.edgecolor':'#777777','pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(8.6,5.3))
for ax,R,panel in zip(axs,[4,8],['a','b']):
 for p in [q for q in points if q['R']==R]:
  x,y=p['x'],p['y'];c=colors[p['method']];empty=p['mu']==0
  ax.errorbar(x['mean'],y['mean'],xerr=x['std'],yerr=y['std'],fmt='none',ecolor=c,elinewidth=.9,capsize=2.3,capthick=.85,alpha=.65,zorder=2)
  ax.plot(x['mean'],y['mean'],marker=markers[p['lambda_index']],ms=6.3,mfc='white' if empty else c,mec=c,mew=1.15,linestyle='none',zorder=4 if empty else 3)
 tv=d['tv_psnr'][str(R)];ax.axhline(tv,color='#606060',ls=(0,(4,3)),lw=1.1,zorder=1)
 ax.text(.985,tv+.065,f'TV  {tv:.2f} dB',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8.5,color='#555555',bbox=dict(facecolor='white',edgecolor='none',pad=1.0,alpha=.9))
 ax.set_title(f'({panel})  {R}× acceleration',fontsize=12,fontweight='semibold',loc='left',pad=12)
 ax.set_xlim(-4,100);ax.set_ylim((26.5,30) if R==4 else (23,26.5))
 ax.set_xticks([0,20,40,60,80,100]);ax.yaxis.set_major_locator(MultipleLocator(.5));ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
 ax.grid(axis='y',color='#E7E7E7',lw=.65,zorder=0);ax.tick_params(length=3,width=.6,color='#777777',pad=5)
 ax.set_xlabel('Exactly zero coefficients (%)',labelpad=9)
 ax.set_ylabel('Validation PSNR (dB)',labelpad=8)
 # Small aligned key below each panel keeps numerical settings outside the data.
 for k,xx in enumerate([.09,.44,.80]):
  n=next(p['n'] for p in points if p['R']==R and p['mu']==1 and p['lambda_index']==k)
  ax.plot(xx-.045,-.245,marker=markers[k],ms=5,linestyle='none',color='#656565',transform=ax.transAxes,clip_on=False)
  ax.text(xx,-.245,f'{lambdas[R][k]:.5g}',transform=ax.transAxes,ha='left',va='center',fontsize=8.2,color='#444444')
 ax.text(-.05,-.245,'λ',transform=ax.transAxes,ha='left',va='center',fontsize=9,color='#555555')
handles=[Line2D([0],[0],marker='o',color=c,linestyle='',mfc=c,ms=6,label=labels[m]) for m,c in colors.items()]+[Line2D([0],[0],color='#606060',ls=(0,(4,3)),lw=1.1,label='Classical TV')]
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.998),ncol=4,frameon=False,handlelength=1.5,handletextpad=.5,columnspacing=1.8,fontsize=9.5)
fig.legend(handles=[Line2D([0],[0],marker='o',linestyle='',color='#555555',mfc='#555555',label='μ = 1',ms=5.5),Line2D([0],[0],marker='o',linestyle='',color='#555555',mfc='white',label='μ = 0',ms=5.5)],loc='lower center',bbox_to_anchor=(.5,.030),ncol=2,frameon=False,handletextpad=.4,columnspacing=1.5,fontsize=9)
fig.subplots_adjust(left=.08,right=.985,bottom=.315,top=.82,wspace=.30)
fig.savefig(ROOT/'figures/mri_l1_results.pdf',bbox_inches='tight',pad_inches=.12)
fig.savefig(ROOT/'figures/mri_l1_results.png',dpi=240,bbox_inches='tight',pad_inches=.12)
