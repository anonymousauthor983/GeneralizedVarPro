"""Rebuild the exact archived train/validation slices from authorized HDF5 files."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import h5py
def normalize_reference(image,size=320):
    a=np.asarray(image,dtype=np.float32)
    if a.ndim!=2 or not np.isfinite(a).all() or a.min()<0:
        raise ValueError('Invalid RSS reference')
    pads=[(max(0,size-n)//2,(max(0,size-n)+1)//2) for n in a.shape]
    a=np.pad(a,pads)
    starts=[(n-size)//2 for n in a.shape]
    a=a[starts[0]:starts[0]+size,starts[1]:starts[1]+size].copy()
    # RMS normalization in the two-channel real/imaginary convention.
    scale=float(np.sqrt(np.square(a,dtype=np.float64).sum()/(2*size*size)))
    if scale<=0:raise ValueError('All-zero RSS reference')
    return (a/scale).astype(np.float32),scale



p=argparse.ArgumentParser();p.add_argument('--raw-root',required=True);p.add_argument('--output',default='data');a=p.parse_args()
root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
m=json.loads(Path(__file__).with_name('data_manifest.json').read_text())
files={}
for f in Path(a.raw_root).rglob('*.h5'):
 if f.stem in files:raise ValueError('Duplicate volume: '+f.stem)
 files[f.stem]=f
for split,info in m['splits'].items():
 target=root/(split+'.npy');rows=info['examples']
 arr=np.lib.format.open_memmap(target,mode='w+',dtype=np.float32,shape=(len(rows),1,320,320))
 for i,row in enumerate(rows):
  with h5py.File(files[row['volume']],'r') as f:image,scale=normalize_reference(f['reconstruction_rss'][row['slice']])
  if not np.isclose(scale,row['scale'],rtol=1e-6,atol=0):raise ValueError('Reference normalization mismatch')
  arr[i,0]=image
 arr.flush();del arr
 h=hashlib.sha256()
 with target.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 if h.hexdigest()!=info['array_sha256']:raise ValueError('Array checksum mismatch for '+split)
 print(split,len(rows),'checksum verified',flush=True)
(root/'manifest.json').write_text(json.dumps(m,indent=2))
