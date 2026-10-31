"""Parameterize the established single-factor controls around frozen train100."""
import copy,math,functools
import numpy as np
import torch
from torch.nn import functional as F
import core
import experiments.docl as D
ORIGINAL_LOSS=D.cosine_order_loss
ARMS=('reverse','zero_margin','fixed_margin','independent_noise')
EPOCH_SINK=None
@functools.lru_cache(maxsize=64)
def fixed_margin(c):
 a=torch.cumprod(1-torch.linspace(1e-4,.02,50,dtype=torch.float32),0);w,s=torch.triu_indices(49,49,1)+1
 actual=c*(a[w]-a[s]);mean=math.fsum(float(v) for v in actual)/1176
 assert abs(mean-float(actual.double().mean()))<1e-14
 return float(torch.tensor(mean,dtype=torch.float32))
def controlled_loss(za,zw,zs,aw,ass,c,method,reference=False):
 a,w,s=[F.normalize(z,p=2,dim=1,eps=1e-8) for z in (za,zw,zs)]
 if reference:
  d=torch.stack([1-torch.sum(a*w,dim=1),1-torch.sum(a*s,dim=1)],1);dw,ds=d[:,0].contiguous(),d[:,1].contiguous()
 else:dw=1-(a*w).sum(1);ds=1-(a*s).sum(1)
 margin=torch.zeros_like(dw) if method=='zero_margin' else torch.full_like(dw,fixed_margin(c)) if method=='fixed_margin' else c*(aw-ass)
 h=torch.clamp_min((ds-dw if method=='reverse' else dw-ds)+margin,0.) if reference else F.relu((ds-dw if method=='reverse' else dw-ds)+margin)
 return h.mean(),dict(violation_rate=(dw>ds).float().mean().detach(),active_rate=(h>0).float().mean().detach(),dweak=dw.mean().detach(),dstrong=ds.mean().detach(),marginmean=margin.mean().detach())
def train100(train,test,seed,batchsize,device,l,m,hidden=(512,512),method='zero_margin',reference=False):
 assert method in ARMS and l>0 and 1<=m<=4
 old=(D.cosine_order_loss,core.cosine_order_loss,core.ordered_contrastive_loss,core._EPOCH_SINK)
 strong=torch.Generator(device=device).manual_seed(seed+4000000)
 def loss(*args,**kw):return controlled_loss(*args,**kw,method=method,reference=reference)
 def independent(model,x,sch,generator,margin_scale,condition):
  assert condition=='actual'
  if reference:
   first=torch.randint(49,(len(x),),generator=generator,device=generator.device);second=torch.randint(48,(len(x),),generator=generator,device=generator.device);second=second+(second>=first).long()
   w=(torch.minimum(first,second)+1).to(x.device);s=(torch.maximum(first,second)+1).to(x.device)
  else:w,s=D.sample_ordered_timesteps(len(x),50,generator,x.device)
  ew=torch.randn(x.shape,dtype=x.dtype,device=generator.device,generator=generator).to(x.device);es=torch.randn(x.shape,dtype=x.dtype,device=strong.device,generator=strong).to(x.device)
  aw,ass=sch.alpha_bar[w],sch.alpha_bar[s];xw=aw.sqrt()[:,None]*x+(1-aw).sqrt()[:,None]*ew;xs=ass.sqrt()[:,None]*x+(1-ass).sqrt()[:,None]*es
  za=model.encode(x,torch.zeros_like(w).float());zw=model.encode(xw,w.float());zs=model.encode(xs,s.float())
  return controlled_loss(za,zw,zs,aw,ass,margin_scale,'independent_noise',True) if reference else ORIGINAL_LOSS(za,zw,zs,aw,ass,margin_scale)
 if method=='independent_noise':core.ordered_contrastive_loss=independent
 else:D.cosine_order_loss=loss;core.cosine_order_loss=loss
 core._EPOCH_SINK=EPOCH_SINK
 try:result=core.train100(train,test,seed,batchsize,device,l,m,hidden,reference and method!='independent_noise')
 finally:D.cosine_order_loss,core.cosine_order_loss,core.ordered_contrastive_loss,core._EPOCH_SINK=old
 model,sch,opt,info=result;info.update(method=method,fixed_margin=fixed_margin(m) if method=='fixed_margin' else None)
 return model,sch,opt,info
def diagnostics(candidate,correct_checkpoint,sch,test,ids,y,seed,c,device):
 idx=np.flatnonzero(y==0);idx=idx[np.argsort(ids[idx],kind='stable')][:128];assert len(idx)>0
 x=torch.from_numpy(np.asarray(test[idx],np.float32)).to(device);correct=copy.deepcopy(candidate);correct.load_state_dict(torch.load(correct_checkpoint,map_location=device,weights_only=True));correct.eval();candidate.eval();arrays={'raw_normal_indices':ids[idx]};out=[]
 for b,off in [('A',22000000),('B',23000000)]:
  gen=torch.Generator(device=device).manual_seed(seed+off);values={k:[] for k in ['uniform','candidate']}
  with torch.no_grad():
   for repeat in range(8):
    w,s=D.sample_ordered_timesteps(len(x),50,gen,device);eps=torch.randn(x.shape,dtype=x.dtype,device=device,generator=gen);aw,ass=sch.alpha_bar[w],sch.alpha_bar[s]
    xw=aw.sqrt()[:,None]*x+(1-aw).sqrt()[:,None]*eps;xs=ass.sqrt()[:,None]*x+(1-ass).sqrt()[:,None]*eps
    for name,model in [('uniform',correct),('candidate',candidate)]:
     a,zw,zs=[F.normalize(z,p=2,dim=1,eps=1e-8) for z in [model.encode(x,torch.zeros_like(w).float()),model.encode(xw,w.float()),model.encode(xs,s.float())]]
     values[name].append(torch.stack([1-(a*zw).sum(1),1-(a*zs).sum(1),c*(aw-ass)],1).cpu().numpy())
  for name,records in values.items():
   v=np.stack(records);arrays[f'{name}_{b}']=v;gap=v[:,:,1].astype(np.float64)-v[:,:,0].astype(np.float64)
   out.append(dict(probe_bank=b,model=name,rows=len(idx),pairs_per_row=8,mean_dstrong_minus_dweak=float(gap.mean()),original_order_violation_rate=float((gap<0).mean()),correct_margin_active_rate=float((gap<v[:,:,2]).mean())))
 return arrays,out
