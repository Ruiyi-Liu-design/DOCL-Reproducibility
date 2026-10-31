"""Only replace DOCL pair selection; retain the original train100 loop and RNG."""
from __future__ import annotations
import hashlib,time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
import core
from experiments.docl import DOCLDenoiser,sample_ordered_timesteps,cosine_order_loss
from vendor.AnoDDAE.src.model import DiffusionScheduler

ARMS=('fixed10_35','mainlinked12','gapcurriculum')
EPOCH_SINK=None

def canonical_bands():
    # Rank the original CPU FP32 schedule in FP64; no dataset or outcomes used.
    a=torch.cumprod(1-torch.linspace(1e-4,.02,50),dim=0).numpy().astype(np.float64)
    pairs=sorted(((w,s) for w in range(1,50) for s in range(w+1,50)),key=lambda p:(a[p[0]]-a[p[1]],p[0],p[1]))
    return np.asarray(pairs,np.int64).reshape(3,392,2)

class PairSampler:
    def __init__(self,method,seed,device,reference=False):
        assert method in (*ARMS,'uniform')
        self.method=method;self.device=torch.device(device);self.reference=reference
        self.generator=torch.Generator(device=device).manual_seed(seed+3000000)
        self.bands=torch.from_numpy(canonical_bands()).to(device)

    def select(self,t_main,epoch,aux_generator):
        # Consume the original two draws even when unused: identical auxiliary
        # Gaussian realizations across all arms and the cached uniform control.
        w,s=sample_ordered_timesteps(len(t_main),50,aux_generator,self.device)
        if self.method=='uniform':return w,s
        if self.reference:
            return self.reference_select(t_main,epoch)
        if self.method=='fixed10_35':
            return torch.full_like(t_main,10),torch.full_like(t_main,35)
        if self.method=='mainlinked12':
            return torch.clamp(t_main-12,min=1),torch.clamp(t_main+12,max=49)
        r=(epoch-21)/79
        u=torch.rand((len(t_main),),device=self.device,generator=self.generator)
        offset=torch.randint(392,(len(t_main),),device=self.device,generator=self.generator)
        band=(u>=(1/6+r/3)).long()+(u>=(.5+r/3)).long()
        pair=self.bands[band,offset]
        return pair[:,0],pair[:,1]

    def reference_select(self,t_main,epoch):
        # Independent branch, literal CPU specification and pair ordering. Random
        # values are drawn on the running device to compare exact actual CUDA.
        if self.method=='fixed10_35':
            w=torch.ones_like(t_main)*10;s=torch.ones_like(t_main)*35
        elif self.method=='mainlinked12':
            w=torch.maximum(t_main-12,torch.ones_like(t_main))
            s=torch.minimum(t_main+12,torch.ones_like(t_main)*49)
        else:
            u=torch.rand((len(t_main),),device=self.device,generator=self.generator)
            offset=torch.randint(392,(len(t_main),),device=self.device,generator=self.generator)
            c1=1/6+(epoch-21)/79/3;c2=c1+1/3
            # Mathematically identical FP thresholds, evaluated by different code.
            # Use the frozen FP32 threshold values to eliminate representational
            # differences in c1+1/3 versus .5+r/3 at an exact boundary.
            c1=float(np.float32(1/6+(epoch-21)/79/3))
            c2=float(np.float32(.5+(epoch-21)/79/3))
            b=torch.where(u<c1,torch.zeros_like(offset),torch.where(u<c2,torch.ones_like(offset),torch.ones_like(offset)*2))
            flat=self.bands.reshape(1176,2);p=flat[b*392+offset];w=p[:,0];s=p[:,1]
        return w,s

def auxiliary(model,x,sch,gen,sampler,t_main,epoch,m,reference=False):
    w,s=sampler.select(t_main,epoch,gen)
    eps=torch.randn(x.shape,dtype=x.dtype,device=gen.device,generator=gen).to(x.device)
    a=sch.alpha_bar.to(device=x.device,dtype=x.dtype);aw,ass=a[w],a[s]
    xw=aw.sqrt().unsqueeze(1)*x+(1-aw).sqrt().unsqueeze(1)*eps
    xs=ass.sqrt().unsqueeze(1)*x+(1-ass).sqrt().unsqueeze(1)*eps
    z0=model.encode(x,torch.zeros_like(w).float());zw=model.encode(xw,w.float());zs=model.encode(xs,s.float())
    loss,stats=cosine_order_loss(z0,zw,zs,aw,ass,m)
    diag=torch.stack((w.double().sum(),s.double().sum(),(aw-ass).double().sum()))
    return loss,stats,diag

def train100(train,test,seed,batchsize,device,l=1.2,m=2.,hidden=(512,512),method='fixed10_35',reference=False):
    device=torch.device(device);core.seed_all(seed)
    model=DOCLDenoiser(train.shape[1],list(hidden),4,activation='lrelu').to(device)
    sch=DiffusionScheduler(50,device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    initial=core.model_digest(model)
    loader=DataLoader(TensorDataset(torch.from_numpy(train),torch.arange(len(train))),batch_size=batchsize,shuffle=True)
    gen=torch.Generator(device=device).manual_seed(seed+1000000)
    sampler=PairSampler(method,seed,device,reference)
    visits=np.zeros(len(train),np.int64);order=hashlib.sha256();trace=[];sampling=[];replay=[];steps=0
    core.synchronize(device);began=time.perf_counter()
    for epoch in range(1,101):
        model.train();tot=torch.zeros(4,device=device,dtype=torch.float64);seen=np.zeros(len(train),bool)
        diag=torch.zeros(3,device=device,dtype=torch.float64)
        for cpu_x,ids in loader:
            ix=ids.numpy();assert not seen[ix].any();seen[ix]=True;visits[ix]+=1;order.update(ix.tobytes())
            x=cpu_x.to(device);t=torch.randint(1,50,(len(x),),device=device).long();xt,_=sch.q_sample(x,t)
            rec=F.mse_loss(model(xt,t.float()),x);loss=rec
            aux=rec.detach()*0;active=aux
            if epoch>20 and l:
                aux,stats,di=auxiliary(model,x,sch,gen,sampler,t,epoch,m,reference);diag+=di.detach()
                active=stats['active_rate'];loss=rec+l*aux
            opt.zero_grad();loss.backward();opt.step();steps+=1
            tot+=torch.stack((rec.detach(),aux.detach(),active,loss.detach())).double()*len(x)
        assert seen.all()
        vals=(tot/len(train)).cpu().tolist();assert np.isfinite(vals).all()
        trace.append(dict(epoch=epoch,rows=len(train),mse=vals[0],docl=vals[1],active=vals[2],loss=vals[3]))
        dv=(diag/len(train)).cpu().tolist()
        sampling.append(dict(epoch=epoch,tweak_mean=dv[0],tstrong_mean=dv[1],alpha_gap_mean=dv[2]))
        if EPOCH_SINK is not None:EPOCH_SINK(dict(**trace[-1],sampling=sampling[-1]))
        if epoch==20:warm=core.model_digest(model);warm_rng=core.main_rng(device)
        if epoch%10==0:
            before=core.main_rng(device);core.replay_evaluation_noise(test,sch,device)
            replay.append(dict(epoch=epoch,before=before,after=core.main_rng(device)))
    core.synchronize(device)
    astep=[int(s['step']) for s in opt.state.values()]
    assert min(astep)==max(astep)==steps==100*len(loader) and (visits==100).all()
    info=dict(epochs=100,steps=steps,rows_per_epoch=len(train),visits_min=int(visits.min()),visits_max=int(visits.max()),
        initial_parameter_hash=initial,warmup_parameter_hash=warm,warmup_rng=warm_rng,
        final_parameter_hash=core.model_digest(model),main_rng_after_training=core.main_rng(device),
        private_final_hash=core.digest_tensor(gen.get_state()),order_hash=order.hexdigest(),
        trace=trace,evaluation_noise_replay=replay,train_seconds=time.perf_counter()-began,
        sampling_trace=sampling,pair_generator_final_hash=core.digest_tensor(sampler.generator.get_state()),method=method)
    return model,sch,opt,info
