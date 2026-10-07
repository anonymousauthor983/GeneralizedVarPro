import argparse,json,time
from pathlib import Path
import torch
from core import *
from data import load
from metrics_ops import cartesian_mask,image_metrics

def main(task,device='cuda',resume=False):
    c=json.loads(Path('config.json').read_text());family=task//9;R=c['accelerations'][family//3];power=c['powers'][family%3];cal=json.loads(Path(f'CALIBRATION_{family}.json').read_text());assert cal['passed']
    method=c['methods'][(task%9)//3];lam=cal['lambdas'][task%3];out=Path('results')/f'r{R}_p{power:g}_{method}_l{task%3}';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(c['seed']);a,selection=load(c)
    m=cartesian_mask(320,R,0,device);vm=cartesian_mask(320,R,100,device)
    net=FeatureNet().to(device);opt=torch.optim.Adam(net.parameters(),lr=c['lr'])
    ordergen=torch.Generator().manual_seed(c['seed']);codes=torch.zeros(1920,16,dtype=torch.float64);history=[];statuses=[];start=time.perf_counter()
    start_epoch=0;resume_record=None
    if resume and (out/'latest.pt').exists():
        checkpoint=torch.load(out/'latest.pt',map_location=device,weights_only=False)
        assert checkpoint['lambda_train']==lam and checkpoint['config']==c
        net.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer'])
        codes=checkpoint['codes'].cpu();start_epoch=checkpoint['epoch']
        for _ in range(start_epoch):torch.randperm(1920,generator=ordergen)
        # Recover progress from the latest matching attempt, without counting failed attempts twice.
        candidates=[]
        for path in Path('logs').glob('*.out'):
            records=[]
            for line in path.read_text(errors='replace').splitlines():
                if not line.startswith('{'):continue
                try:r=json.loads(line)
                except json.JSONDecodeError:continue
                if (r.get('method'),r.get('power'),r.get('R'),r.get('lambda_train'))==(method,power,R,lam) and r.get('step',0)<=start_epoch*480:records.append(r)
            if records and records[-1]['step']==start_epoch*480:candidates.append((path.stat().st_mtime,records,str(path)))
        if candidates:
            _,history,source=max(candidates,key=lambda z:z[0])
            start-=history[-1]['seconds']
        else:source=None
        resume_record=dict(checkpoint_epoch=start_epoch,history_source=source,
                           historical_training_solver_states_available=False,
                           train_time_includes_recovered_progress_seconds=bool(candidates))
        print(json.dumps(dict(resume=resume_record)),flush=True)
    for epoch in range(start_epoch,c['epochs']):
        order=torch.randperm(1920,generator=ordergen)
        for bi in range(0,1920,4):
            ids=order[bi:bi+4];x=a['train'][ids].to(device);y=acquisition(x,m);B=net(features_input(y));old=codes[ids].to(device)
            if method=='projected':
                H,b=gram(B,y,m);w,st=solve_codes(H,b,lam,old,power=power);used=implicit_codes(H,b,w,lam,st,power=power)
                rec=synth(B,used);loss=.5*(rec-x).square().mean()
            else:
                H,b=gram(B,y,m,x,c['mu'])
                if method=='alternating':
                    w,st=solve_codes(H,b,lam,old,power=power);used=w.detach()
                else:
                    with torch.no_grad():w=prox_step(H,b,old,lam,power=power)
                    used=old;st=[]
                rec=synth(B,used);loss=(.5*(rec-x).square().flatten(1).sum(1)+c['mu']*.5*(acquisition(rec,m)-y).abs().square().flatten(1).sum(1)+lam*used.abs().pow(power).sum(1)).mean()/(320**2)
            opt.zero_grad();loss.backward();assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())
            opt.step();codes[ids]=w.detach().cpu();statuses.extend(st)
            progress=dict(method=method,power=power,R=R,lambda_train=lam,epoch=epoch+1,step=epoch*480+bi//4+1,loss=float(loss.detach()),seconds=time.perf_counter()-start)
            history.append(progress);(out/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,codes=codes,config=c,lambda_train=lam),out/'latest.tmp')
        (out/'latest.tmp').replace(out/'latest.pt')
    if device=='cuda':torch.cuda.synchronize()
    train_seconds=time.perf_counter()-start
    # Same measurement-only inference for ALL methods. All grid values reported.
    results=[];net.eval()
    for lt in [lam]:
        metrics=[];states=[];total=time.perf_counter()
        with torch.no_grad():
            for bi in range(0,480,4):
                x=a['val'][bi:bi+4].to(device);y=acquisition(x,vm);B=net(features_input(y));H,b=gram(B,y,vm);w,st=solve_codes(H,b,lt,power=power)
                rec=synth(B,w);mm=image_metrics(rec,x)
                threshold=c['active_threshold_relative']*w.abs().max(1,keepdim=True)[0].clamp_min(1.)
                mm['active_codes']=(w.abs()>threshold).double().sum(1)
                mm['exact_zero_percent']=(w==0).double().mean(1)*100
                mm['measurement_mse']=(acquisition(rec,vm)-y).abs().square().flatten(1).mean(1)
                metrics.extend([{k:float(v[j]) for k,v in mm.items()} for j in range(len(x))]);states.extend(st)
        results.append(dict(lambda_test=lt,validation={k:sum(r[k] for r in metrics)/480 for k in metrics[0]},per_image=metrics,solver_states=states,evaluation_seconds=time.perf_counter()-total))
    selected=min(results,key=lambda r:r['validation']['nmse'])
    data=dict(method=method,power=power,R=R,lambda_train=lam,seed=c['seed'],mu=c['mu'],D=16,epochs=8,updates=3840,train_seconds=train_seconds,all_inference_results=results,selected_on_validation=selected,training_history=history,training_solver_states=statuses,resume=resume_record)
    (out/'result.tmp').write_text(json.dumps(data,indent=2))
    (out/'result.tmp').replace(out/'result.json')
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--task',type=int,required=True);ap.add_argument('--resume',action='store_true');a=ap.parse_args();main(a.task,resume=a.resume)
