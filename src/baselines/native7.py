"""Seven fixed baseline adapters. No metrics, search, or epoch selection.

Official files stay byte-identical in sources/. Only data/epoch/output routing
and explicitly recorded missing settings are supplied here.
"""
from pathlib import Path
import copy,hashlib,importlib.util,json,math,random,sys,time,types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent
S=ROOT/'sources'
def load(name,path):
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module
def namespace(name,path):
    m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ah(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
def configure():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def rng(device):
    r={'cpu':ah(torch.get_rng_state().numpy())}
    if str(device).startswith('cuda'):r['cuda']=ah(torch.cuda.get_rng_state(device).cpu().numpy())
    return r
def native_ddae():
    pkg=ROOT/'sources/AnoDDAE'
    namespace('native_anoddae',pkg);namespace('native_anoddae.src',pkg/'src')
    return load('native_anoddae.src.model',pkg/'src/model.py')
def official(method):
    if method=='RDP':return load('author_rdp',S/'billhhh__RDP/RDP-Anomaly/model.py')
    if method=='MCM':
        namespace('Model',S/'JXYin24__MCM/Model')
        return (load('Model.Model',S/'JXYin24__MCM/Model/Model.py'),
                load('Model.Loss',S/'JXYin24__MCM/Model/Loss.py'),
                load('Model.Score',S/'JXYin24__MCM/Model/Score.py'))
    if method=='DRL':
        # ipdb is an unused debug import, not part of execution mathematics.
        sys.modules.setdefault('ipdb',types.ModuleType('ipdb'))
        return load('author_drl',S/'HangtingYe__DRL/PrototypeNet/Model.py')
    if method=='Disent':return load('author_disent',S/'yjnanan__Disent-AD/network/dis_net.py')
    if method=='NeuTraL':
        b=S/'boschresearch__NeuTraL-AD/models'
        return (load('author_neutral_tabs',b/'TabNets.py'),load('author_neutral_net',b/'NeutralAD.py'),load('author_neutral_loss',b/'Losses.py'))
    raise ValueError(method)
class State:
    pass
def make(method,d,cfg,device):
    device=torch.device(device);s=State();s.method=method;s.cfg=copy.deepcopy(cfg);s.device=device
    s.scheduler=None;s.criterion=None;s.lr_scheduler=None
    if method in ['DAE','DDAE-C']:
        native=native_ddae()
        s.net=native.DiffusionDenoiser(d,[512,512],time_emb_dim=0 if method=='DAE' else 4).to(device)
        if method=='DDAE-C':s.scheduler=native.DiffusionScheduler(50,device)
        s.opt=torch.optim.Adam(s.net.parameters(),lr=cfg['lr'])
    elif method=='RDP':
        s.rdp=official(method).RDP_Model(d,50,USE_GPU=device.type=='cuda',LR=cfg['lr'],dropout_r=.1)
        s.net=s.rdp.r_net;s.opt=s.rdp.r_net_optim
    elif method=='MCM':
        model,loss,score=official(method)
        conf=dict(cfg,data_dim=d,device=str(device))
        s.net=model.MCM(conf).to(device);s.criterion=loss.LossFunction(conf).to(device);s.score=score.ScoreFunction(conf).to(device)
        s.opt=torch.optim.Adam(s.net.parameters(),lr=cfg['lr'],weight_decay=1e-5)
        s.lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(s.opt,gamma=.98)
    elif method=='DRL':
        conf=dict(cfg,data_dim=d,device=str(device))
        s.net=official(method).PN(conf).to(device)
        s.opt=torch.optim.Adam(s.net.parameters(),lr=cfg['lr'],weight_decay=1e-5)
        s.lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(s.opt,gamma=.98)
    elif method=='NeuTraL':
        tab,net,loss=official(method)
        conf=dict(cfg,device=str(device),trans_hdim=d)
        s.net=net.TabNeutralAD(tab.TabNets(),d,conf).to(device);s.criterion=loss.DCL(temperature=.1)
        s.opt=torch.optim.Adam(s.net.parameters(),lr=cfg['lr'],weight_decay=1e-5)
        s.lr_scheduler=torch.optim.lr_scheduler.StepLR(s.opt,step_size=200,gamma=.5)
    elif method=='Disent':
        s.net=official(method).DisNet(dim=cfg['hidden_dim'],att_dim=cfg['patch_size']).to(device)
        s.opt=torch.optim.Adam(s.net.parameters(),lr=cfg['lr'])
    else:raise ValueError(method)
    return s
def encode(net,x,t):
    emb=net.sine_cosine_transform_timesteps(t)
    return net.network[:-1](torch.cat([x,emb],dim=1))
def contrastive_terms(z0,zp,zt,t):
    positive=(z0-zp).square().sum(dim=1)
    negative=F.relu(1+t.float()/50-torch.linalg.vector_norm(z0-zt,dim=1)).square()
    return .5*(positive.mean()+negative.mean())
def drl_loss(net,x,cfg):
    loss=net(x).mean()
    h=net.encoder(x);recon=net.decoder(h)
    loss=loss+cfg['input_info_ratio']*(1-F.cosine_similarity(recon,x,dim=-1)).mean()
    coeff=net.phi(x)
    selected=np.random.choice(len(x),int(len(x)*.8),replace=False)
    # Author formula is undefined for zero selected rows. The empty pair set
    # contributes zero; no samples are dropped or duplicated to fill a batch.
    coeff=coeff[selected]
    if len(coeff):
        similarity=coeff@coeff.T
        denominator=coeff.square().sum(-1,keepdim=True).sqrt()@coeff.T.square().sum(0,keepdim=True).sqrt()
        similarity=similarity/denominator
        penalty=((1-torch.eye(len(coeff),device=x.device))*similarity).sum()/len(coeff)/len(coeff)
        loss=loss+cfg['cl_ratio']*penalty
    return loss
def patch(x,cfg):return x.unfold(1,cfg['patch_size'],cfg['patch_size']-cfg['overlap'])
def one_update(s,x,all_train,private,epoch):
    if s.method=='RDP':
        value=s.rdp.train_model(x.detach().cpu().numpy(),epoch)
        return float(value)
    if s.method=='DAE':
        noisy=x+s.cfg['sigma']*torch.randn_like(x)
        loss=F.mse_loss(s.net(noisy,torch.zeros(len(x),device=x.device)),x)
    elif s.method=='DDAE-C':
        t=torch.randint(1,50,(len(x),),device=x.device).long();xt,_=s.scheduler.q_sample(x,t)
        rec=F.mse_loss(s.net(xt,t.float()),x);loss=rec
        if s.cfg['alpha']:
            ix=torch.randint(len(all_train),(len(x),),generator=private)
            positive=all_train[ix].to(x.device)
            z0=encode(s.net,x,t);zp=encode(s.net,positive,t);zt=encode(s.net,xt,t)
            aux=contrastive_terms(z0,zp,zt,t)
            loss=(1-s.cfg['alpha'])*rec+s.cfg['alpha']*aux
    elif s.method=='MCM':
        prediction,_,masks=s.net(x);loss,_,_=s.criterion(x,prediction,masks)
    elif s.method=='DRL':loss=drl_loss(s.net,x,s.cfg)
    elif s.method=='NeuTraL':loss=s.criterion(s.net(x)).mean()
    elif s.method=='Disent':
        rec,dis=s.net(patch(x,s.cfg));loss=rec+dis
    else:raise ValueError(s.method)
    if not torch.isfinite(loss):raise FloatingPointError('NONFINITE_TRAINING_LOSS')
    s.opt.zero_grad();loss.backward();s.opt.step()
    return float(loss.detach())
def replay_eval(test,sch,device):
    with torch.no_grad():
        for t in range(1,50):
            for start in range(0,len(test),8192):
                x=torch.as_tensor(test[start:start+8192],device=device)
                sch.q_sample(x,torch.full((len(x),),t,device=device,dtype=torch.int64))
def train(method,train_x,test_x,cfg,seed,device,callback=None):
    seed_all(seed);s=make(method,train_x.shape[1],cfg,device)
    x=torch.from_numpy(train_x);private=torch.Generator().manual_seed(seed+1_000_000)
    shuffle=method not in ['MCM','DRL']
    loader=DataLoader(TensorDataset(x,torch.arange(len(x))),batch_size=cfg['batch_size'],shuffle=shuffle,drop_last=False)
    visits=np.zeros(len(x),np.int64);history=[];replay=[];step=0;start=time.perf_counter();order=hashlib.sha256()
    for epoch in range(1,101):
        s.net.train();seen=np.zeros(len(x),bool);total=0.
        for batch,ids in loader:
            ix=ids.numpy();assert not seen[ix].any();seen[ix]=True;visits[ix]+=1;order.update(ix.tobytes())
            total+=one_update(s,batch.to(s.device),x,private,epoch)*len(batch);step+=1
        assert seen.all()
        if s.lr_scheduler:s.lr_scheduler.step()
        if method=='DDAE-C' and epoch%10==0:
            before=rng(s.device);replay_eval(test_x,s.scheduler,s.device)
            replay.append(dict(epoch=epoch,before=before,after=rng(s.device)))
        rec=dict(epoch=epoch,rows=len(x),optimizer_steps=step,loss=total/len(x))
        if not math.isfinite(rec['loss']):raise FloatingPointError('NONFINITE_EPOCH_LOSS')
        history.append(rec)
        if callback:callback(rec)
    assert (visits==100).all() and step==100*len(loader)
    s.info=dict(epochs=100,optimizer_steps=step,train_rows=len(x),visits_min=int(visits.min()),visits_max=int(visits.max()),train_order_sha256=order.hexdigest(),trace=history,evaluation_rng_replay=replay,main_rng_after_training=rng(s.device),train_seconds=time.perf_counter()-start)
    return s
def score(s,test,noise_seed):
    s.net.eval();start=time.perf_counter();parts=[];before=rng(s.device)
    if s.method=='DDAE-C':
        gen=torch.Generator(device=s.device).manual_seed(noise_seed)
        result=np.zeros(len(test),np.float32)
        with torch.no_grad():
            for t in range(1,50):
                for starti in range(0,len(test),8192):
                    x=torch.as_tensor(test[starti:starti+8192],device=s.device)
                    eps=torch.randn(x.shape,device=s.device,generator=gen);a=s.scheduler.alpha_bar[t]
                    xt=a.sqrt()*x+(1-a).sqrt()*eps
                    tt=torch.full((len(x),),t,device=s.device,dtype=torch.int64)
                    result[starti:starti+len(x)]+=torch.linalg.vector_norm(x-s.net(xt,tt),dim=1).cpu().numpy()
    elif s.method=='RDP':
        # The author evaluator uses a random partner permutation. Private A/B
        # state makes this repeatable without disturbing the training stream.
        saved=np.random.get_state();np.random.seed(noise_seed)
        try:
            with torch.no_grad():
                for i in range(0,len(test),s.cfg['batch_size']):parts.append(s.rdp.eval_model(test[i:i+s.cfg['batch_size']]))
        finally:np.random.set_state(saved)
        result=np.concatenate(parts).reshape(-1)
    else:
        with torch.no_grad():
            for i in range(0,len(test),s.cfg['batch_size']):
                x=torch.as_tensor(test[i:i+s.cfg['batch_size']],device=s.device)
                if s.method=='DAE':value=torch.linalg.vector_norm(x-s.net(x,torch.zeros(len(x),device=s.device)),dim=1)
                elif s.method=='MCM':value=s.score(x,s.net(x)[0])
                elif s.method=='DRL':
                    h=s.net.encoder(x);hhat=s.net.phi(x)@s.net.prototype
                    value=F.mse_loss(h,hhat,reduction='none').mean(-1)
                elif s.method=='NeuTraL':value=s.criterion(s.net(x),eval=True)
                elif s.method=='Disent':value=s.net(patch(x,s.cfg))
                parts.append(value.cpu().numpy().reshape(-1))
        result=np.concatenate(parts)
    assert before==rng(s.device),'Scoring leaked into main Torch RNG'
    assert result.shape==(len(test),) and np.isfinite(result).all(),'NONFINITE_OR_INCOMPLETE_NATIVE_SCORE'
    return result,time.perf_counter()-start
def checkpoint(s):
    result=dict(model=s.net.state_dict(),optimizer=s.opt.state_dict(),config=s.cfg,training=s.info)
    if s.method=='RDP':result['fixed_target']=s.rdp.r_target_net.state_dict()
    if s.scheduler:result['alpha_bar']=s.scheduler.alpha_bar
    return result
