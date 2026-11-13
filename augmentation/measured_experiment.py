"""Measured NW-UCLA augmentation experiment; historical Inception-v3 protocol."""
import os, random, re, json, time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import Inception_V3_Weights, inception_v3
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED=42; SEGMENTS=4; IMAGE_SIZE=224; BATCH=2; WORKERS=2; EPOCHS=12; LR=.001
OUT=Path('/kaggle/working/augmentation_results'); OUT.mkdir(parents=True,exist_ok=True)
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def stamp(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def log(message):
    line=f'{stamp()} {message}'; print(line,flush=True)
    with (OUT/'training.log').open('a',encoding='utf8') as f: f.write(line+'\n')
def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def inventory():
    roots=[p for p in Path('/kaggle/input').rglob('multiview_action_videos') if (p/'a01').is_dir()]
    if len(roots)!=1: raise RuntimeError(f'Expected one dataset root: {roots}')
    groups=defaultdict(dict); pattern=re.compile(r'v(\d+)_s(\d+)_e(\d+)\.avi$',re.I)
    for p in sorted(roots[0].rglob('*.avi')):
        m=pattern.fullmatch(p.name)
        if not m: continue
        a=int(p.parent.name[1:]); v,s,e=map(int,m.groups()); key=(a,s,e)
        if v in groups[key]: raise RuntimeError(f'duplicate {key} {v}')
        groups[key][v]=str(p)
    actions=sorted({key[0] for key in groups}); mapping={a:i for i,a in enumerate(actions)}
    samples=[dict(action=mapping[a],subject=s,event=e,paths=[v[1],v[2],v[3]]) for (a,s,e),v in sorted(groups.items()) if all(i in v for i in (1,2,3))]
    if len(samples)!=418 or len(actions)!=10: raise RuntimeError(f'Unexpected inventory: {len(samples)} groups, {actions}')
    return samples,actions
MEAN=np.array([.485,.456,.406],np.float32); STD=np.array([.229,.224,.225],np.float32)
def read_segments(path,train):
    cap=cv2.VideoCapture(path)
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or n<=0: raise RuntimeError(f'Unreadable video {path}')
        ids=[]
        for i in range(SEGMENTS):
            lo=min(n-1,i*n//SEGMENTS); hi=min(n-1,max(lo,(i+1)*n//SEGMENTS-1)); ids.append(random.randint(lo,hi) if train else (lo+hi)//2)
        # One draw per clip; preserve ordered motion across all its segments.
        flip=train and random.random()<.5
        color=train and random.random()<.5
        contrast=random.uniform(.9,1.1) if color else 1.; brightness=random.uniform(-8,8) if color else 0.
        result=[]
        for idx in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,im=cap.read()
            if not ok: raise RuntimeError(f'Cannot decode {path}, frame {idx}')
            im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(IMAGE_SIZE,IMAGE_SIZE))
            if flip: im=np.ascontiguousarray(im[:,::-1])
            if color: im=np.clip(im.astype(np.float32)*contrast+brightness,0,255).astype(np.uint8)
            result.append(torch.from_numpy(((im.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1).copy()))
        return torch.stack(result)
    finally: cap.release()
class MultiViewDataset(Dataset):
    def __init__(self,samples,train): self.samples=samples; self.train=train
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]; return torch.stack([read_segments(p,self.train) for p in s['paths']]),torch.tensor(s['action'])
class InceptionBN(nn.Module):
    """Shared Inception-v3/BN trunk, returning the final spatial feature map."""
    def __init__(self):
        # Current torchvision requires aux_logits=True when loading ImageNet weights.
        # The auxiliary head is not called because this class explicitly forwards the trunk.
        super().__init__(); m=inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1,aux_logits=True)
        for n in ["Conv2d_1a_3x3","Conv2d_2a_3x3","Conv2d_2b_3x3","maxpool1","Conv2d_3b_1x1","Conv2d_4a_3x3","maxpool2","Mixed_5b","Mixed_5c","Mixed_5d","Mixed_6a","Mixed_6b","Mixed_6c","Mixed_6d","Mixed_6e","Mixed_7a","Mixed_7b","Mixed_7c"]: setattr(self,n,getattr(m,n))
    def forward(self,x):
        x=self.Conv2d_1a_3x3(x); x=self.Conv2d_2a_3x3(x); x=self.Conv2d_2b_3x3(x); x=self.maxpool1(x); x=self.Conv2d_3b_1x1(x); x=self.Conv2d_4a_3x3(x); x=self.maxpool2(x)
        x=self.Mixed_5b(x); x=self.Mixed_5c(x); x=self.Mixed_5d(x); x=self.Mixed_6a(x); x=self.Mixed_6b(x); x=self.Mixed_6c(x); x=self.Mixed_6d(x); x=self.Mixed_6e(x); x=self.Mixed_7a(x); x=self.Mixed_7b(x); return self.Mixed_7c(x)

class MultiBranchTSNGRU(nn.Module):
    def __init__(self, branches):
        super().__init__(); self.branches=branches; self.shared_inception_bn=InceptionBN()
        self.view_refine=nn.ModuleList([nn.Sequential(nn.Conv2d(2048,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d(1)) for _ in range(branches)])
        self.shared_gru=nn.GRU(256,128,batch_first=True); self.classifier=nn.Sequential(nn.Dropout(.5),nn.Linear(128*branches,10))
    def forward(self,x):
        b,v,k,c,h,w=x.shape; z=self.shared_inception_bn(x.reshape(b*v*k,c,h,w)); z=z.reshape(b,v,k,2048,z.shape[-2],z.shape[-1]); out=[]
        for j in range(v):
            q=self.view_refine[j](z[:,j].reshape(b*k,2048,z.shape[-2],z.shape[-1])).flatten(1).reshape(b,k,256)
            out.append(self.shared_gru(q)[0][:,-1])
        return self.classifier(torch.cat(out,1))


def evaluate(model,loader):
    model.eval(); y=[]; pred=[]; loss=0
    with torch.no_grad():
        for x,t in loader:
            o=model(x.to(DEVICE)); loss+=nn.functional.cross_entropy(o,t.to(DEVICE),reduction='sum').item(); y.extend(t.tolist()); pred.extend(o.argmax(1).cpu().tolist())
    return {'loss':loss/len(y),'accuracy':accuracy_score(y,pred),'macro_f1':f1_score(y,pred,labels=list(range(10)),average='macro',zero_division=0),'support':len(y)},y,pred

def fit(name,train,validation=None):
    seed_all(); model=MultiBranchTSNGRU(2).to(DEVICE)
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=.9,weight_decay=5e-4)
    scheduler=torch.optim.lr_scheduler.MultiStepLR(opt,[4,9],gamma=.1)
    loader=DataLoader(MultiViewDataset(train,True),batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=True)
    val_loader=DataLoader(MultiViewDataset(validation,False),batch_size=BATCH,num_workers=WORKERS,pin_memory=True) if validation else None
    records=[]
    for epoch in range(1,EPOCHS+1):
        model.train(); total=0; correct=0; count=0; start=time.time()
        for x,y in loader:
            x=x.to(DEVICE); y=y.to(DEVICE); opt.zero_grad(); o=model(x); loss=nn.functional.cross_entropy(o,y); loss.backward(); opt.step()
            total+=loss.item()*len(y); correct+=(o.argmax(1)==y).sum().item(); count+=len(y)
        record={'timestamp_utc':stamp(),'phase':name,'epoch':epoch,'train_loss':total/count,'train_accuracy':correct/count,'learning_rate':opt.param_groups[0]['lr'],'seconds':time.time()-start}
        if val_loader:
            metrics,_,_=evaluate(model,val_loader); record.update({'validation_loss':metrics['loss'],'validation_accuracy':metrics['accuracy'],'validation_macro_f1':metrics['macro_f1']})
        records.append(record); log(json.dumps(record)); scheduler.step()
    pd.DataFrame(records).to_csv(OUT/f'{name}_history.csv',index=False)
    torch.save({'model':model.state_dict(),'seed':SEED,'phase':name,'epochs':EPOCHS,'actions':ACTIONS},OUT/f'{name}_checkpoint.pt')
    return model,records

def main():
    global ACTIONS
    samples,ACTIONS=inventory(); train=[dict(s,paths=s['paths'][:2]) for s in samples]
    test=[dict(s,paths=[s['paths'][2],s['paths'][2]]) for s in samples]
    dev_train=[s for s in train if s['subject']!=10]; val=[s for s in train if s['subject']==10]
    (OUT/'split_manifest.json').write_text(json.dumps({'train':train,'validation_subject':10,'test':test,'actions':ACTIONS},indent=2))
    log(json.dumps({'event':'start','device':DEVICE,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'seed':SEED,'epochs':EPOCHS,'train_pairs':len(train),'validation_count':len(val),'test_count':len(test),'augmentation':'clip-consistent flip p=.5 and contrast/brightness p=.5'}))
    _,history=fit('development',dev_train,val)
    model,_=fit('refit',train)
    metrics,y,pred=evaluate(model,DataLoader(MultiViewDataset(test,False),batch_size=BATCH,num_workers=WORKERS,pin_memory=True))
    metrics.update({'timestamp_utc':stamp(),'experiment':'augmented_inception_v3_cross_view_12_to_3','epochs':EPOCHS,'seed':SEED})
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2)); (OUT/'evaluation.log').write_text(json.dumps(metrics,indent=2)+'\n')
    pd.DataFrame({'true':y,'predicted':pred,'action_id':[ACTIONS[i] for i in y]}).to_csv(OUT/'test_predictions.csv',index=False)
    cm=confusion_matrix(y,pred,labels=list(range(10))); pd.DataFrame(cm,index=ACTIONS,columns=ACTIONS).to_csv(OUT/'confusion_matrix.csv')
    plt.figure(figsize=(9,7)); sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=ACTIONS,yticklabels=ACTIONS); plt.xlabel('Predicted action'); plt.ylabel('True action'); plt.tight_layout(); plt.savefig(OUT/'confusion_matrix.png',dpi=150); plt.close()
    h=pd.DataFrame(history); fig,ax=plt.subplots(1,2,figsize=(11,4)); ax[0].plot(h.epoch,h.train_loss,label='train'); ax[0].plot(h.epoch,h.validation_loss,label='validation'); ax[0].legend(); ax[0].set_title('Loss'); ax[1].plot(h.epoch,h.train_accuracy,label='train'); ax[1].plot(h.epoch,h.validation_accuracy,label='validation'); ax[1].legend(); ax[1].set_title('Accuracy'); fig.tight_layout(); fig.savefig(OUT/'learning_curves.png',dpi=150); plt.close(fig)
    log('COMPLETE '+json.dumps(metrics))
if __name__=='__main__': main()
