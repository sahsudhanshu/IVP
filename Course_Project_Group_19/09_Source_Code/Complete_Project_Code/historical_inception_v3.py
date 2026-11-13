"""Genuine NW-UCLA RGB reproduction: EDA -> Multi-branch TSN + GRU -> evaluation.

Implements the available-dataset portion of Bui & Nguyen (2020): shared
Inception-BN encoder, per-view 3x3 refinement, shared GRU TSN consensus, and
late feature concatenation. MicaHandGesture is intentionally not substituted.
"""
import json, os, random, re, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception_V3_Weights, inception_v3

SEED = 42; SEGMENTS = 4; IMAGE_SIZE = 224; BATCH = 2; WORKERS = 2
# Actual training configuration. Increase to 80 to match the paper's schedule.
EPOCHS = int(os.getenv("EPOCHS", "12")); LR = 1e-3
OUT = Path("/kaggle/working/basepaper_nwucla_results"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def locate_root():
    bases = list(Path("/kaggle/input").rglob("multiview_action_videos"))
    bases = [p for p in bases if p.is_dir() and (p / "a01").is_dir()]
    if len(bases) != 1: raise RuntimeError(f"Expected one NW-UCLA RGB root, found: {bases}")
    return bases[0]

ROOT = locate_root()
RX = re.compile(r"v(\d+)_s(\d+)_e(\d+)\.avi$", re.I)
CLASS_MAP = {}

def inventory():
    rows=[]
    for p in sorted(ROOT.rglob("*.avi")):
        m=RX.search(p.name)
        if not m: continue
        cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        rows.append(dict(path=str(p), action=int(p.parent.name[1:]), view=int(m.group(1)), subject=int(m.group(2)), event=int(m.group(3)), frames=n, fps=fps, width=w, height=h))
    df=pd.DataFrame(rows); df.to_csv(OUT/"video_inventory.csv",index=False); return df

def savefig(name): plt.tight_layout(); plt.savefig(OUT/name,dpi=160,bbox_inches="tight"); plt.close()

def eda(df):
    sns.set_theme(style="whitegrid")
    for col,title,file in [("action","Videos per action","eda_action_distribution.png"),("view","Videos per view","eda_view_distribution.png"),("subject","Videos per subject","eda_subject_distribution.png")]:
        plt.figure(figsize=(9,4)); sns.countplot(data=df,x=col,color="#377eb8"); plt.title(title); savefig(file)
    plt.figure(figsize=(9,4)); sns.histplot(df.frames,bins=30,color="#e41a1c"); plt.title("Video frame-count distribution"); plt.xlabel("frames"); savefig("eda_frame_distribution.png")
    plt.figure(figsize=(8,4)); sns.countplot(data=df,x="action",hue="view",palette="Set2"); plt.title("Action-view coverage"); savefig("eda_action_by_view.png")
    picks=df.groupby("action",group_keys=False).apply(lambda x:x.sample(1,random_state=SEED))
    fig,axes=plt.subplots(2,5,figsize=(15,6));
    for ax,(_,r) in zip(axes.ravel(),picks.iterrows()):
        cap=cv2.VideoCapture(r.path); cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,r.frames//2)); ok,im=cap.read(); cap.release()
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB) if ok else np.zeros((20,20,3),dtype=np.uint8)); ax.set_title(f"a{r.action:02d} v{r.view} s{r.subject}"); ax.axis("off")
    savefig("eda_rgb_samples.png")
    return {"videos":len(df),"actions":df.action.nunique(),"subjects":df.subject.nunique(),"views":df.view.nunique(),"frames_median":int(df.frames.median()),"root":str(ROOT)}

def grouped(df):
    g=defaultdict(dict)
    for r in df.itertuples(): g[(r.action,r.subject,r.event)][r.view]=r.path
    result=[]
    for (a,s,e),views in g.items():
        if all(v in views for v in (1,2,3)): result.append(dict(action=CLASS_MAP[a],subject=s,event=e,paths=[views[1],views[2],views[3]]))
    return result

MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
def read_segments(path, train):
    cap=cv2.VideoCapture(path); n=max(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))); ids=[]
    for i in range(SEGMENTS):
        lo=i*n//SEGMENTS; hi=max(lo,(i+1)*n//SEGMENTS-1); ids.append(random.randint(lo,hi) if train else (lo+hi)//2)
    frames=[]
    for idx in ids:
        cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,im=cap.read()
        if not ok: im=np.zeros((IMAGE_SIZE,IMAGE_SIZE,3),np.uint8)
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB); im=cv2.resize(im,(IMAGE_SIZE,IMAGE_SIZE))
        if train and random.random()<.5: im=np.ascontiguousarray(im[:,::-1])
        frames.append(torch.from_numpy(((im.astype(np.float32)/255.-MEAN)/STD).transpose(2,0,1)))
    cap.release(); return torch.stack(frames)

class MultiViewDataset(Dataset):
    def __init__(self, samples, views, train): self.samples=samples; self.views=views; self.train=train
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]; return torch.stack([read_segments(s["paths"][v-1],self.train) for v in self.views]), torch.tensor(s["action"]), s["subject"]

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

def run_experiment(name, train_s, test_s, views, labels):
    model=MultiBranchTSNGRU(len(views)).to(DEVICE)
    # Fine-tune the proposed architecture end-to-end; pretraining only initializes Inception-BN.
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=.9,weight_decay=5e-4); sch=torch.optim.lr_scheduler.MultiStepLR(opt,[max(1,EPOCHS*3//8),max(2,EPOCHS*6//8)],gamma=.1); lossfn=nn.CrossEntropyLoss()
    tr=DataLoader(MultiViewDataset(train_s,views,True),batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=True); te=DataLoader(MultiViewDataset(test_s,views,False),batch_size=BATCH,num_workers=WORKERS,pin_memory=True)
    hist=[]
    for ep in range(1,EPOCHS+1):
        model.train(); losses=[]; yy=[]; pp=[]
        for x,y,_ in tr:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); o=model(x); l=lossfn(o,y); l.backward(); opt.step(); losses.append(l.item()); yy+=y.cpu().tolist(); pp+=o.argmax(1).detach().cpu().tolist()
        model.eval(); ty=[]; tp=[]
        with torch.no_grad():
            for x,y,_ in te: o=model(x.to(DEVICE)); ty+=y.tolist(); tp+=o.argmax(1).cpu().tolist()
        rec=dict(epoch=ep,train_loss=float(np.mean(losses)),train_acc=accuracy_score(yy,pp),test_acc=accuracy_score(ty,tp),lr=opt.param_groups[0]["lr"]); hist.append(rec); print(name,rec,flush=True); sch.step()
    pd.DataFrame(hist).to_csv(OUT/f"{name}_history.csv",index=False)
    torch.save({"model":model.state_dict(),"views":views,"epochs":EPOCHS},OUT/f"{name}_checkpoint.pt")
    fig,ax=plt.subplots(1,2,figsize=(11,4)); h=pd.DataFrame(hist); ax[0].plot(h.epoch,h.train_loss); ax[0].set_title(f"{name}: loss"); ax[1].plot(h.epoch,h.train_acc,label="train"); ax[1].plot(h.epoch,h.test_acc,label="test"); ax[1].legend(); ax[1].set_title(f"{name}: accuracy"); savefig(f"{name}_learning_curves.png")
    cm=confusion_matrix(ty,tp,labels=list(range(10))); plt.figure(figsize=(8,7)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels); plt.title(f"{name}: test confusion matrix"); plt.xlabel("predicted"); plt.ylabel("true"); savefig(f"{name}_confusion_matrix.png")
    return dict(experiment=name,test_accuracy=accuracy_score(ty,tp),macro_f1=f1_score(ty,tp,average="macro",zero_division=0),test_samples=len(ty))

def main():
    global CLASS_MAP
    df=inventory(); CLASS_MAP={a:i for i,a in enumerate(sorted(df.action.unique()))}; meta=eda(df); samples=grouped(df); print("EDA",meta,"class_map",CLASS_MAP,"three_view_groups",len(samples),flush=True)
    # Paper protocol implementations on available NW-UCLA: leave-one-subject-out fold and all 3 cross-view splits.
    labels=[f"a{i:02d}" for i in range(1,11)]; results=[]
    held=1; results.append(run_experiment("cross_subject_holdout_s01",[s for s in samples if s["subject"]!=held],[s for s in samples if s["subject"]==held],[1,2,3],labels))
    for target in (1,2,3):
        source=[v for v in (1,2,3) if v!=target]
        # Training uses two source views. Test duplicates unseen target as prescribed in the paper.
        train=[dict(s,paths=[s["paths"][source[0]-1],s["paths"][source[1]-1]]) for s in samples]
        test=[dict(s,paths=[s["paths"][target-1],s["paths"][target-1]]) for s in samples]
        results.append(run_experiment(f"cross_view_{{{source[0]},{source[1]}}}_to_{target}",train,test,[1,2],labels))
    r=pd.DataFrame(results); r.to_csv(OUT/"evaluation_summary.csv",index=False); plt.figure(figsize=(8,4)); sns.barplot(data=r,x="experiment",y="test_accuracy",color="#4daf4a"); plt.xticks(rotation=25,ha="right"); plt.ylim(0,1); plt.title("Actual NW-UCLA evaluation accuracy"); savefig("evaluation_summary.png")
    (OUT/"conclusions.md").write_text("# Conclusions\n\nThis run uses only the public NW-UCLA RGB dataset; MicaHandGesture was not available and was not replaced. Metrics are actual outputs from this run, not copied from the paper. The script implements the paper's multi-branch RGB TSN+GRU design and available cross-subject/cross-view protocols.\n")
    print("COMPLETE",json.dumps({"device":DEVICE,"output":str(OUT),"results":results,"eda":meta},indent=2),flush=True)

if __name__=="__main__": main()
