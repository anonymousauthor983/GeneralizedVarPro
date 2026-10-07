"""Export exact Gaussian-mixture level-set samples as genuine TikZ paths.
The only approximation is contour extraction on a documented regular grid;
there are no embedded raster images. Compile figures/tikz/contours_*.tex.
"""
from pathlib import Path
import json
import numpy as np
import contourpy
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reproducibility/results';T=ROOT/'figures/tikz';T.mkdir(parents=True,exist_ok=True)
cfg=json.loads((R/'config.json').read_text())
summary=json.loads((R/'summary.json').read_text())
scores={seed:np.mean([r['ise'] for r in summary if r['seed']==seed and r['method'] in ['joint','palm','varpro']]) for seed in cfg['seeds']}
ordered=sorted(cfg['seeds'],key=lambda seed:(scores[seed],seed))
representative_seed=ordered[(len(ordered)-1)//2]
d=np.load(R/f"seed_{representative_seed}.npz")
sigma=cfg['sigma'];truth_t=np.array(cfg['true_centers']);truth_w=np.array(cfg['true_weights'])
levels=[.15,.4,.8,1.2,1.6,2.2,2.8]
axis=np.linspace(-1.8,1.8,721);X,Y=np.meshgrid(axis,axis)
def density(points,t,w):
    ans=np.zeros(points.shape[:-1]);w=w/w.sum()
    for a,b in zip(t,w):ans+=b*np.exp(-np.sum((points-a)**2,axis=-1)/(2*sigma*sigma))/(2*np.pi*sigma*sigma)
    return ans
xy=np.stack([X,Y],axis=-1)
def simplify(v,tol=1e-4):
    """Ramer--Douglas--Peucker, measured in data coordinates."""
    if len(v)<=2:return v
    ab=v[-1]-v[0];norm=np.dot(ab,ab)
    if norm==0:dist=np.linalg.norm(v-v[0],axis=1)
    else:
        u=np.clip((v-v[0])@ab/norm,0,1)
        dist=np.linalg.norm(v-(v[0]+u[:,None]*ab),axis=1)
    j=int(np.argmax(dist))
    if dist[j]<=tol:return v[[0,-1]]
    return np.vstack([simplify(v[:j+1],tol)[:-1],simplify(v[j:],tol)])
checks={};paths={}
for m in ['truth','joint','palm','varpro','prox1','prox10','grid']:
    if m=='truth':t,w=truth_t,truth_w
    elif m=='grid':t,w=d['grid_theta'],d['grid_w']
    else:t,w=d[m+'_theta'][-1],d[m+'_w'][-1]
    dens=density(xy,t,w);gen=contourpy.contour_generator(x=axis,y=axis,z=dens,name='serial')
    paths[m]=[];errors=[]
    for level in levels:
        loops=gen.lines(level);entry=[]
        for v in loops:
            assert np.linalg.norm(v[0]-v[-1])<1e-10,'Open contour: enlarge plotting domain'
            # Sub-pixel polyline simplification for portable TeX memory use.
            v=simplify(v)
            checkpts=np.vstack([v,(v[:-1]+v[1:])/2])
            errors.extend(np.abs(density(checkpts,t,w)-level).tolist())
            entry.append(v)
        paths[m].append(entry)
    checks[m]={'max_density_level_residual_vertices_and_midpoints':max(errors,default=0),'max_sampled_density':float(dens.max()),'active':int(np.sum(w>cfg['support_threshold']))}
# The numerical paths are written directly as TikZ coordinates in normalized
# axis units, avoiding any shell-escape/gnuplot requirement at compile time.
def curve(v):return ' -- '.join(f'(axis cs:{a:.8f},{b:.8f})' for a,b in v[:-1])+' -- cycle'
for m in paths:
    tex=[]
    if m!='truth':
        for j,loops in enumerate(paths[m]):
            if loops:tex.append(r'\path[fill=sketchblue'+str(j)+r',draw=sketchline,line width=.3pt,even odd rule] '+' '.join(curve(v) for v in loops)+';\n')
    for j,loops in enumerate(paths[m] if m=='truth' else []):
        for v in loops:
            style='draw=black!55,densely dashed,line width=.3pt' if m=='truth' else 'draw=sketchline,line width=.3pt'
            tex.append(r'\path['+style+'] '+curve(v)+';\n')
    if m!='truth':
        t=d['grid_theta'] if m=='grid' else d[m+'_theta'][-1];w=d['grid_w'] if m=='grid' else d[m+'_w'][-1]
        for a,b in zip(t,w/w.sum()):
            if b*w.sum()>cfg['support_threshold']:
                # Marker area (not radius) proportional to normalized mass,
                # with a small visibility floor, explicitly documented.
                radius=np.sqrt(1+8*b)
                tex.append(r'\draw[draw=sketchorange,fill=none,line width=.65pt] '+f'(axis cs:{a[0]:.8f},{a[1]:.8f}) circle[radius={radius:.4f}pt];\n')
    (T/(m+'_paths.tex')).write_text(''.join(tex))
labels={'joint':'Joint PG','palm':'PALM','varpro':'VarPro','prox1':'Alt-1','prox10':'Alt-10','grid':'Fixed grid'}
header=r'''\documentclass[tikz,border=3pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{sketchline}{HTML}{176B99}
\definecolor{sketchorange}{HTML}{D55E00}
'''
# Sequential Blues palette, identical across panels.
for j,h in enumerate(['EDF5FC','D9EAF5','B5D7EC','86BCDC','529AC8','2879B0','08519C']):header+=r'\definecolor{sketchblue'+str(j)+'}{HTML}{'+h+'}\n'
header+=r'''\begin{document}
\begin{tikzpicture}
'''
for variant,methods,cols in [('main',['joint','palm','varpro'],3),('all',['joint','palm','varpro','prox1','prox10','grid'],3)]:
    rows=len(methods)//cols
    tex=header+r'\begin{groupplot}[group style={group size=3 by '+str(rows)+r''',horizontal sep=0.45cm,vertical sep=1.2cm},
width=5.15cm,height=5.15cm,scale only axis,axis equal image,
xmin=-1.8,xmax=1.8,ymin=-1.8,ymax=1.8,
xtick={-1,0,1},ytick={-1,0,1},tick label style={font=\small},
title style={font=\normalsize},label style={font=\small},
axis line style={black!60},tick style={black!60},
axis background/.style={fill=white},clip=true]
'''
    for index,m in enumerate(methods):
        opts=[f'title={{{labels[m]} ($s={checks[m]["active"]}$)}}']
        if index%cols==0:opts.append(r'ylabel={$x_2$}')
        else:opts.append('yticklabels={,,}')
        if index//cols==rows-1:opts.append(r'xlabel={$x_1$}')
        tex+=r'\nextgroupplot['+','.join(opts)+']\n'+r'\input{'+m+'_paths.tex}\n'+r'\input{truth_paths.tex}'+'\n'
        tex+=r'\addplot[only marks,mark=+,mark size=2.2pt,black,line width=.65pt] coordinates {'+' '.join(f'({a},{b})' for a,b in truth_t)+'};\n'
    tex+=r'\end{groupplot}'+'\n'
    # One compact shared key; contour values also supplied explicitly in caption.
    tex+=r'\node[anchor=north,font=\small,align=center] at ([yshift=-1.05cm]group c2r'+str(rows)+r'.south) {Blue: fitted density \quad Dashed: population \quad $+$: true means \quad Orange: active means};'+'\n'
    tex+=r'\end{tikzpicture}'+'\n'+r'\end{document}'+'\n'
    (T/f'contours_{variant}.tex').write_text(tex)
meta={'polyline_simplification_tolerance':1e-4,'grid_size':721,'grid_spacing':float(axis[1]-axis[0]),'domain':[-1.8,1.8],'levels':levels,'seed':representative_seed,'selection':'Lower median trial ranked by mean final ISE across Joint PG, PALM and VarPro; no selection on component count','contourpy_version':contourpy.__version__,'checks':checks,'rendering':'TikZ paths and PGFPlots axes; no bitmap; level sets sampled from analytic Gaussian mixture'}
(T/'contour_validation.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta,indent=2))
