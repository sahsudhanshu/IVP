"""NW-UCLA inventory, paired-view split, clip-consistent preprocessing."""
from collections import defaultdict
from pathlib import Path
import random, re
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

ACTIONS = [1,2,3,4,5,6,8,9,11,12]
ACTION_NAMES = ['Pick up with one hand','Pick up with two hands','Drop trash','Walk around','Sit down','Stand up','Put on a jacket','Take off a jacket','Throw','Carry']
MEAN = np.array([.485,.456,.406], np.float32)
STD = np.array([.229,.224,.225], np.float32)

def inventory(root):
    root=Path(root)
    if not root.is_dir(): raise FileNotFoundError(f'Dataset directory not found: {root}')
    roots=([root] if root.name=='multiview_action_videos' else list(root.rglob('multiview_action_videos')))
    if len(roots)!=1: raise ValueError(f'Expected one multiview_action_videos directory, found {roots}')
    pattern=re.compile(r'v(\d+)_s(\d+)_e(\d+)\.avi$',re.I)
    groups=defaultdict(dict)
    for p in sorted(roots[0].rglob('*.avi')):
        match=pattern.fullmatch(p.name)
        if not match or not re.fullmatch(r'a\d+',p.parent.name, re.I): continue
        a=int(p.parent.name[1:]); v,s,e=map(int,match.groups())
        if a not in ACTIONS: continue
        key=(a,s,e)
        if v in groups[key]: raise ValueError(f'Duplicate video for {key}, view {v}')
        groups[key][v]=str(p.resolve())
    rows=[{'action_id':a,'label':ACTIONS.index(a),'subject':s,'event':e,'paths':[views[i] for i in (1,2,3)]}
          for (a,s,e),views in sorted(groups.items()) if all(i in views for i in (1,2,3))]
    if not rows: raise ValueError('No complete three-view groups were found')
    return rows

def segment_indices(n, count, train):
    if n<=0 or count<=0: raise ValueError('Frame and segment counts must be positive')
    indices=[]
    for i in range(count):
        lo=min(n-1, i*n//count); hi=min(n-1,max(lo,(i+1)*n//count-1))
        indices.append(random.randint(lo,hi) if train else (lo+hi)//2)
    return indices

def read_clip(path, frames=8, size=224, train=False):
    cap=cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {path}')
        count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); indices=segment_indices(count,frames,train)
        flip=train and random.random()<.5
        jitter=train and random.random()<.5
        contrast=random.uniform(.9,1.1) if jitter else 1.0
        brightness=random.uniform(-8,8) if jitter else 0.0
        output=[]
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES,index); ok,image=cap.read()
            if not ok: raise RuntimeError(f'Cannot decode frame {index} in {path}')
            image=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),(size,size))
            if flip: image=np.ascontiguousarray(image[:,::-1])
            if jitter: image=np.clip(image.astype(np.float32)*contrast+brightness,0,255).astype(np.uint8)
            output.append(torch.from_numpy(((image.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1).copy()))
        return torch.stack(output)
    finally: cap.release()

class VideoGroups(Dataset):
    def __init__(self, groups, split, frames, size):
        if split not in ('train','validation','test'): raise ValueError(split)
        self.groups=groups; self.split=split; self.frames=frames; self.size=size
    def __len__(self): return len(self.groups)
    def __getitem__(self,index):
        group=self.groups[index]; views=(0,1) if self.split in ('train','validation') else (2,)
        clips=torch.stack([read_clip(group['paths'][v],self.frames,self.size,self.split=='train') for v in views])
        return clips,torch.tensor(group['label'],dtype=torch.long)
