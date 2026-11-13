"""NW-UCLA RGB approximation: EDA -> Multi-branch TSN + GRU -> evaluation.

Uses a pretrained BN-Inception trunk and the published TSN preprocessing.
The unspecified branch widths and GRU size are implementation choices.
MicaHandGesture is not substituted.
"""
import hashlib, json, os, random, re, time
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


SEED = 42; SEGMENTS = 4; IMAGE_SIZE = 224; BATCH = 2; WORKERS = 2
# Paper schedule; EPOCHS can be overridden for smoke tests.
EPOCHS = int(os.getenv("EPOCHS", "80")); LR = 1e-3
VAL_SUBJECT = int(os.getenv("VAL_SUBJECT", "0"))
EXPERIMENT = os.getenv("EXPERIMENT", "all")
RESUME = os.getenv("RESUME", "0") == "1"
OUT = Path(os.getenv("OUTPUT_DIR", "/kaggle/working/basepaper_nwucla_results")); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def locate_root():
    if os.getenv("DATA_ROOT"):
        root = Path(os.environ["DATA_ROOT"])
        if not (root / "a01").is_dir(): raise RuntimeError(f"Invalid DATA_ROOT: {root}")
        return root
    bases = list(Path("/kaggle/input").rglob("multiview_action_videos"))
    bases = [p for p in bases if p.is_dir() and (p / "a01").is_dir()]
    if len(bases) != 1: raise RuntimeError(f"Expected one NW-UCLA RGB root, found: {bases}")
    return bases[0]

ROOT = None
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
    df=pd.DataFrame(rows); df.to_csv(OUT/"video_inventory.csv",index=False)
    # Check against the lists shipped with the RGB archive, not against a guessed total.
    missing=[]; unlisted=[]
    for listing in sorted(ROOT.glob("a*/videos.txt")):
        expected={line.strip() for line in listing.read_text(errors="replace").splitlines() if line.strip()}
        actual={p.name for p in listing.parent.glob("*.avi")}
        missing.extend(str(listing.parent/name) for name in sorted(expected-actual))
        unlisted.extend(str(listing.parent/name) for name in sorted(actual-expected))
    audit={"video_count":len(df),"missing_from_shipped_lists":missing,"unlisted_videos":unlisted,
           "pairing_assumption":"Matching action, subject and event identify the same recording; verify independently."}
    (OUT/"dataset_audit.json").write_text(json.dumps(audit,indent=2))
    if missing or unlisted: raise RuntimeError("Dataset video lists do not match files; see dataset_audit.json")
    return df

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

def grouped(df, required_views):
    """Keep every action with the views needed for this split; reject duplicates."""
    g=defaultdict(dict)
    for r in df.itertuples():
        key=(r.action,r.subject,r.event)
        if r.view in g[key]:
            raise ValueError(f"Duplicate view {r.view} for {key}")
        g[key][r.view]=r.path
    result=[]
    for (a,subject,event),views in sorted(g.items()):
        if all(v in views for v in required_views):
            result.append(dict(action=CLASS_MAP[a], subject=subject, event=event,
                               paths=[views.get(v) for v in (1,2,3)]))
    return result

BGR_MEAN=np.array([104,117,128],np.float32)
def read_segments(path, train):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise RuntimeError(f"Cannot open video: {path}")
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0: raise RuntimeError(f"Invalid frame count for {path}: {n}")
        ids=[]
        for i in range(SEGMENTS):
            lo=min(n-1, i*n//SEGMENTS)
            hi=min(n-1, max(lo, (i+1)*n//SEGMENTS-1))
            ids.append(random.randint(lo,hi) if train else (lo+hi)//2)
        frames=[]
        for idx in ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,im=cap.read()
            if not ok: raise RuntimeError(f"Cannot decode frame {idx} in {path}")
            # Preserve aspect ratio; deterministic center crop, no augmentation.
            h,w=im.shape[:2]; scale=256/min(h,w)
            im=cv2.resize(im,(max(IMAGE_SIZE,round(w*scale)),max(IMAGE_SIZE,round(h*scale))))
            h,w=im.shape[:2]; y=(h-IMAGE_SIZE)//2; x=(w-IMAGE_SIZE)//2
            im=im[y:y+IMAGE_SIZE,x:x+IMAGE_SIZE].astype(np.float32)-BGR_MEAN
            frames.append(torch.from_numpy(im.transpose(2,0,1).copy()))
        return torch.stack(frames)
    finally:
        cap.release()

class MultiViewDataset(Dataset):
    def __init__(self, samples, views, train): self.samples=samples; self.views=views; self.train=train
    def __len__(self): return len(self.samples)
    def __getitem__(self,i):
        s=self.samples[i]; return torch.stack([read_segments(s["paths"][v-1],self.train) for v in self.views]), torch.tensor(s["action"]), s["subject"]

class BNInceptionTrunk(nn.Module):
    """Cadene BN-Inception, pretrained on ImageNet in BGR 0-255 space."""
    def __init__(self):
        super().__init__()
        try:
            import pretrainedmodels
        except ImportError as exc:
            raise RuntimeError("Install pretrainedmodels==0.7.4 in the Kaggle notebook") from exc
        weights=os.getenv("BN_WEIGHTS")
        self.model=pretrainedmodels.bninception(num_classes=1000, pretrained=None if weights else "imagenet")
        if weights:
            state=torch.load(weights, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state,strict=True)
        self.model.last_linear=nn.Identity()
        # TSN partial BN: retain first BN statistics; freeze remaining backbone BNs.
        self.frozen_bn=[]
        for bn in [m for m in self.model.modules() if isinstance(m,nn.BatchNorm2d)][1:]:
            bn.requires_grad_(False); self.frozen_bn.append(bn)

    def train(self, mode=True):
        super().train(mode)
        for bn in self.frozen_bn: bn.eval()
        return self

    def forward(self,x):
        return self.model.features(x)

class MultiBranchTSNGRU(nn.Module):
    def __init__(self, branches):
        super().__init__(); self.branches=branches; self.shared_inception_bn=BNInceptionTrunk()
        self.view_refine=nn.ModuleList([nn.Sequential(nn.Conv2d(1024,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(),nn.AdaptiveAvgPool2d(1)) for _ in range(branches)])
        self.shared_gru=nn.GRU(256,128,batch_first=True); self.classifier=nn.Sequential(nn.Dropout(.5),nn.Linear(128*branches,10))
    def forward(self,x):
        b,v,k,c,h,w=x.shape; z=self.shared_inception_bn(x.reshape(b*v*k,c,h,w)); z=z.reshape(b,v,k,1024,z.shape[-2],z.shape[-1]); out=[]
        for j in range(v):
            q=self.view_refine[j](z[:,j].reshape(b*k,1024,z.shape[-2],z.shape[-1])).flatten(1).reshape(b,k,256)
            out.append(self.shared_gru(q)[0][:,-1])
        return self.classifier(torch.cat(out,1))

def run_experiment(name, train_s, test_s, views, labels):
    if not train_s or not test_s: raise ValueError(f"{name}: empty training or test split")
    # Each experiment has the same initialization whether run alone or in a batch.
    experiment_seed=SEED+int.from_bytes(hashlib.sha256(name.encode()).digest()[:4],"big")
    random.seed(experiment_seed); np.random.seed(experiment_seed % (2**32)); torch.manual_seed(experiment_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(experiment_seed)
    model=MultiBranchTSNGRU(len(views)).to(DEVICE)
    # Fine-tune the proposed architecture end-to-end; pretraining initializes BN-Inception.
    opt=torch.optim.SGD(model.parameters(),lr=LR,momentum=.9,weight_decay=5e-4); sch=torch.optim.lr_scheduler.MultiStepLR(opt,[30,60],gamma=.1); lossfn=nn.CrossEntropyLoss()
    tr=DataLoader(MultiViewDataset(train_s,views,True),batch_size=BATCH,shuffle=True,num_workers=WORKERS,pin_memory=True); te=DataLoader(MultiViewDataset(test_s,views,False),batch_size=BATCH,num_workers=WORKERS,pin_memory=True)
    manifest={"train":train_s,"evaluation":test_s,"views":views,"class_map":CLASS_MAP}
    (OUT/f"{name}_split.json").write_text(json.dumps(manifest,indent=2))
    config={"epochs":EPOCHS,"views":views,"segments":SEGMENTS,"batch":BATCH,
            "lr":LR,"seed":experiment_seed,"class_map":CLASS_MAP,"manifest":manifest}
    checkpoint=OUT/f"{name}_resume.pt"
    hist=[]; start=1
    if RESUME and checkpoint.exists():
        saved=torch.load(checkpoint,map_location=DEVICE,weights_only=False)
        if saved["config"] != config: raise RuntimeError(f"Resume settings or split changed for {name}")
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"]); sch.load_state_dict(saved["scheduler"])
        hist=saved["history"]; start=saved["epoch"]+1
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available() and saved["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved["cuda_rng"]])
    for ep in range(start,EPOCHS+1):
        model.train(); losses=[]; yy=[]; pp=[]
        for x,y,_ in tr:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(); o=model(x); l=lossfn(o,y); l.backward(); opt.step(); losses.append(l.item()); yy+=y.cpu().tolist(); pp+=o.argmax(1).detach().cpu().tolist()
        rec=dict(epoch=ep,train_loss=float(np.mean(losses)),train_acc=accuracy_score(yy,pp),lr=opt.param_groups[0]["lr"]); hist.append(rec); print(name,rec,flush=True); sch.step()
        tmp=checkpoint.with_suffix(".tmp")
        torch.save({"config":config,"epoch":ep,"model":model.state_dict(),"optimizer":opt.state_dict(),
                    "scheduler":sch.state_dict(),"history":hist,"python_rng":random.getstate(),
                    "numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
                    "cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},tmp)
        tmp.replace(checkpoint)
    # The held-out test split is evaluated once, after the fixed training schedule.
    model.eval(); ty=[]; tp=[]
    with torch.no_grad():
        for x,y,_ in te: o=model(x.to(DEVICE)); ty+=y.tolist(); tp+=o.argmax(1).cpu().tolist()
    records=[]
    for sample,y,pred in zip(test_s,ty,tp):
        records.append(dict(subject=sample["subject"],event=sample["event"],paths=sample["paths"],true=y,predicted=pred))
    (OUT/f"{name}_predictions.json").write_text(json.dumps(records,indent=2))
    pd.DataFrame(hist).to_csv(OUT/f"{name}_history.csv",index=False)
    torch.save({"model":model.state_dict(),"views":views,"epochs":EPOCHS,"class_map":CLASS_MAP,"backbone":"BN-Inception","segments":SEGMENTS,"input":"BGR 0-255, mean 104/117/128"},OUT/f"{name}_checkpoint.pt")
    fig,ax=plt.subplots(1,2,figsize=(11,4)); h=pd.DataFrame(hist); ax[0].plot(h.epoch,h.train_loss); ax[0].set_title(f"{name}: loss"); ax[1].plot(h.epoch,h.train_acc,label="train"); ax[1].legend(); ax[1].set_title(f"{name}: accuracy"); savefig(f"{name}_learning_curves.png")
    cm=confusion_matrix(ty,tp,labels=list(range(10))); plt.figure(figsize=(8,7)); sns.heatmap(cm,annot=True,fmt="d",cmap="Blues",xticklabels=labels,yticklabels=labels); plt.title(f"{name}: test confusion matrix"); plt.xlabel("predicted"); plt.ylabel("true"); savefig(f"{name}_confusion_matrix.png")
    return dict(experiment=name,test_accuracy=accuracy_score(ty,tp),macro_f1=f1_score(ty,tp,labels=list(range(10)),average="macro",zero_division=0),test_samples=len(ty))

def main():
    global CLASS_MAP, ROOT
    ROOT=locate_root()
    df=inventory()
    if df.empty: raise RuntimeError("No matching videos found")
    if df.frames.le(0).any(): raise RuntimeError("Invalid video frame counts; see video_inventory.csv")
    CLASS_MAP={int(a):i for i,a in enumerate(sorted(df.action.unique()))}
    if len(CLASS_MAP)!=10: raise RuntimeError(f"Expected 10 classes, got {CLASS_MAP}")
    (OUT/"class_map.json").write_text(json.dumps(CLASS_MAP,indent=2))
    meta=eda(df); samples=grouped(df,(1,2,3)); print("EDA",meta,"class_map",CLASS_MAP,"three_view_groups",len(samples),flush=True)
    # A validation subject is drawn only from source training subjects.
    # Development mode never evaluates the outer test set.
    labels=[f"a{i:02d}" for i in CLASS_MAP]; results=[]
    def run_split(name, train, test, views):
        if EXPERIMENT not in ("all",name): return
        if VAL_SUBJECT:
            if not any(s["subject"]==VAL_SUBJECT for s in train):
                print(f"Skipping {name}: validation subject absent from source training data",flush=True); return
            validation=[s for s in train if s["subject"]==VAL_SUBJECT]
            train=[s for s in train if s["subject"]!=VAL_SUBJECT]
            name+=f"_validation_s{VAL_SUBJECT:02d}"
            test=validation
        results.append(run_experiment(name,train,test,views,labels))
    for held in sorted(df.subject.unique()):
        run_split(f"cross_subject_holdout_s{held:02d}",[s for s in samples if s["subject"]!=held],[s for s in samples if s["subject"]==held],[1,2,3])
    for target in (1,2,3):
        source=[v for v in (1,2,3) if v!=target]
        train=[dict(s,paths=[s["paths"][source[0]-1],s["paths"][source[1]-1]]) for s in grouped(df,source)]
        test=[dict(s,paths=[s["paths"][target-1],s["paths"][target-1]]) for s in grouped(df,(target,))]
        print(f"cross-view {source} to {target}: {len(train)} training pairs, {len(test)} test videos",flush=True)
        run_split(f"cross_view_{{{source[0]},{source[1]}}}_to_{target}",train,test,[1,2])
    if not results: raise RuntimeError(f"No experiments selected: {EXPERIMENT}")
    r=pd.DataFrame(results); r.to_csv(OUT/(("validation_summary" if VAL_SUBJECT else "evaluation_summary") + ("" if EXPERIMENT == "all" else "_" + EXPERIMENT) + ".csv"),index=False); plt.figure(figsize=(8,4)); sns.barplot(data=r,x="experiment",y="test_accuracy",color="#4daf4a"); plt.xticks(rotation=25,ha="right"); plt.ylim(0,1); plt.title("Actual NW-UCLA evaluation accuracy"); savefig("evaluation_summary.png")
    (OUT/"conclusions.md").write_text("# Conclusions\n\nThis run uses only the public NW-UCLA RGB dataset; MicaHandGesture was not available and was not replaced. Metrics are actual outputs from this run, not copied from the paper. BN-Inception and partial BN are used. Branch widths and GRU dimensions are assumptions; event pairing and dataset completeness remain to be independently verified.\n")
    print("COMPLETE",json.dumps({"device":DEVICE,"output":str(OUT),"results":results,"eda":meta},indent=2),flush=True)

if __name__=="__main__": main()
