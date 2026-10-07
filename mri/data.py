import hashlib,json
from pathlib import Path
import torch
import numpy as np

def select(manifest,split,nvol,per):
    groups={}
    for i,e in enumerate(manifest['splits'][split]['examples']):groups.setdefault(e['volume'],[]).append(i)
    ids=[]
    for v,seq in list(groups.items())[:nvol]:
        mid=len(seq)//2;ids.extend(seq[mid-per//2:mid+per//2])
    assert len(ids)==nvol*per
    return ids

def load(c):
    root=Path(c['data']);manifest=json.loads((root/'manifest.json').read_text());arrays={};selections={}
    for s in ['train','val']:
        ids=list(range(len(manifest['splits'][s]['examples'])));file=root/(s+'.npy');h=hashlib.sha256()
        with file.open('rb') as f:
            for z in iter(lambda:f.read(8*1024*1024),b''):h.update(z)
        assert h.hexdigest()==manifest['splits'][s]['array_sha256']
        arrays[s]=torch.tensor(np.array(np.load(file,mmap_mode='r')[ids],copy=True),dtype=torch.float64)
        if arrays[s].ndim==3:arrays[s]=arrays[s][:,None]
        selections[s]=[dict(index=i,**manifest['splits'][s]['examples'][i]) for i in ids]
    assert not ({e['volume'] for e in selections['train']}&{e['volume'] for e in selections['val']})
    return arrays,selections
