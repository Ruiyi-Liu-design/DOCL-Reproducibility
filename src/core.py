from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib,time
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from threadpoolctl import threadpool_limits
from experiments.docl import DOCLDenoiser,ordered_contrastive_loss,cosine_order_loss
from vendor.AnoDDAE.src.model import DiffusionScheduler
_EPOCH_SINK=None


def digest_tensor(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def model_digest(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def main_rng(device):
    value = {'cpu': screen.digest_tensor(torch.get_rng_state())}
    if device.type == 'cuda':
        value['cuda'] = screen.digest_tensor(torch.cuda.get_rng_state(device))
    return value

def replay_evaluation_noise(test, scheduler, device):
    """Stock predict's randn_like calls, no updates or stochastic forward layers."""
    with torch.no_grad():
        for t in range(1, scheduler.num_timesteps):
            for start in range(0, len(test), 8192):
                x = torch.from_numpy(test[start:start+8192]).to(device)
                ts = torch.full((len(x),), t, device=device, dtype=torch.int64)
                scheduler.q_sample(x, ts)

def configure():
    threadpool_limits(limits=2)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def reconstruct(model, scheduler, test, noise_seed, device):
    x = torch.from_numpy(test).to(device)
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    R = torch.zeros(len(x), device=device)
    bands = torch.zeros((3, len(x)), device=device)
    band_steps = np.array_split(np.arange(1, 50), 3)
    which = {int(t): i for i, steps in enumerate(band_steps) for t in steps}
    screen.synchronize(device)
    began = time.perf_counter()
    with torch.no_grad():
        for t in range(1, 50):
            for start in range(0, len(x), 8192):
                batch = x[start:start+8192]
                eps = torch.randn(batch.shape, device=device, generator=generator)
                a = scheduler.alpha_bar[t]
                xt = a.sqrt()*batch+(1-a).sqrt()*eps
                ts = torch.full((len(batch),), t, device=device, dtype=torch.int64)
                error = (batch-model(xt, ts)).norm(dim=1)
                R[start:start+len(batch)] += error
                bands[which[t], start:start+len(batch)] += error
    screen.synchronize(device)
    return R.cpu().numpy(), bands.cpu().numpy(), time.perf_counter()-began, screen.digest_tensor(generator.get_state())

def train100(train,test,seed,batchsize,device,l,m,hidden=(512,512),reference=False):
    device=torch.device(device);utility.seed_all(seed)
    model=DOCLDenoiser(train.shape[1],list(hidden),4,activation='lrelu').to(device)
    sch=DiffusionScheduler(50,device);opt=torch.optim.Adam(model.parameters(),lr=.001)
    initial=utility.model_digest(model)
    loader=DataLoader(TensorDataset(torch.from_numpy(train),torch.arange(len(train))),batch_size=batchsize,shuffle=True)
    gen=torch.Generator(device=device).manual_seed(seed+1000000)
    visits=np.zeros(len(train),np.int64);order=hashlib.sha256();trace=[];replay=[];steps=0
    utility.synchronize(device);began=time.perf_counter()
    for epoch in range(1,101):
        model.train();tot=torch.zeros(4,device=device,dtype=torch.float64);seen=np.zeros(len(train),bool)
        for cpu_x,ids in loader:
            ix=ids.numpy();assert not seen[ix].any();seen[ix]=True;visits[ix]+=1;order.update(ix.tobytes())
            x=cpu_x.to(device);t=torch.randint(1,50,(len(x),),device=device).long();xt,_=sch.q_sample(x,t)
            rec=F.mse_loss(model(xt,t.float()),x);loss=rec
            aux=rec.detach()*0;active=aux
            if epoch>20 and l:
                if reference:
                    # Independently use original view construction and replace only the scalar margin.
                    from experiments.docl import sample_ordered_timesteps
                    tw,ts=sample_ordered_timesteps(len(x),50,gen,device)
                    eps=torch.randn(x.shape,device=device,dtype=x.dtype,generator=gen)
                    aw,ass=sch.alpha_bar[tw],sch.alpha_bar[ts]
                    z0=model.encode(x,torch.zeros_like(tw).float())
                    zw=model.encode(aw.sqrt()[:,None]*x+(1-aw).sqrt()[:,None]*eps,tw.float())
                    zs=model.encode(ass.sqrt()[:,None]*x+(1-ass).sqrt()[:,None]*eps,ts.float())
                    aux,stats=cosine_order_loss(z0,zw,zs,aw,ass,m)
                else:
                    aux,stats=ordered_contrastive_loss(model,x,sch,gen,margin_scale=m,condition='actual')
                active=stats['active_rate'];loss=rec+l*aux
            opt.zero_grad();loss.backward();opt.step();steps+=1
            tot+=torch.stack((rec.detach(),aux.detach(),active,loss.detach())).double()*len(x)
        assert seen.all()
        vals=(tot/len(train)).cpu().tolist();assert np.isfinite(vals).all()
        trace.append(dict(epoch=epoch,rows=len(train),mse=vals[0],docl=vals[1],active=vals[2],loss=vals[3]))
        if _EPOCH_SINK is not None: _EPOCH_SINK(trace[-1])
        if epoch==20: warm=utility.model_digest(model);warm_rng=native.main_rng(device)
        if epoch%10==0:
            before=native.main_rng(device);native.replay_evaluation_noise(test,sch,device)
            replay.append(dict(epoch=epoch,before=before,after=native.main_rng(device)))
    utility.synchronize(device)
    astep=[int(s['step']) for s in opt.state.values()]
    assert min(astep)==max(astep)==steps==100*len(loader) and (visits==100).all()
    info=dict(epochs=100,steps=steps,rows_per_epoch=len(train),visits_min=int(visits.min()),visits_max=int(visits.max()),
        initial_parameter_hash=initial,warmup_parameter_hash=warm,warmup_rng=warm_rng,
        final_parameter_hash=utility.model_digest(model),main_rng_after_training=native.main_rng(device),
        private_final_hash=utility.digest_tensor(gen.get_state()),order_hash=order.hexdigest(),
        trace=trace,evaluation_noise_replay=replay,train_seconds=time.perf_counter()-began)
    return model,sch,opt,info

def eq_state(a,b):
    if isinstance(a,torch.Tensor): return torch.equal(a,b)
    if isinstance(a,dict): return a.keys()==b.keys() and all(eq_state(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(eq_state(x,y) for x,y in zip(a,b))
    return a==b

def independent_ap(y,s):
    order=np.argsort(-s,kind='stable');v=s[order];z=y[order].astype(np.int64)
    ends=np.r_[np.flatnonzero(v[1:]!=v[:-1]),len(v)-1]
    tp=np.cumsum(z,dtype=np.int64)[ends];dt=np.diff(np.r_[0,tp])
    return float(np.sum(dt*(tp/(ends+1)),dtype=np.float64)/sum(z))

utility=SimpleNamespace(seed_all=seed_all,model_digest=model_digest,synchronize=synchronize,digest_tensor=digest_tensor)
screen=utility
native=SimpleNamespace(main_rng=main_rng,replay_evaluation_noise=replay_evaluation_noise)
