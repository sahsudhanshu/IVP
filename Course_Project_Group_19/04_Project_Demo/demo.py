"""Simple single-video NW-UCLA action recognition demo."""
from pathlib import Path
import argparse, threading, sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import inception_v3

HERE=Path(__file__).resolve().parent
CHECKPOINT=HERE.parent/'12_Trained_Models'/'refit_checkpoint.pt'
ACTIONS={1:'Pick up with one hand',2:'Pick up with two hands',3:'Drop trash',4:'Walk around',5:'Sit down',6:'Stand up',8:'Put on a jacket',9:'Take off a jacket',11:'Throw',12:'Carry'}
MEAN=np.array([.485,.456,.406],np.float32); STD=np.array([.229,.224,.225],np.float32)
class InceptionBN(nn.Module):
    """Shared Inception-v3/BN trunk, returning the final spatial feature map."""
    def __init__(self):
        # Current torchvision requires aux_logits=True when loading ImageNet weights.
        # The auxiliary head is not called because this class explicitly forwards the trunk.
        super().__init__(); m=inception_v3(weights=None,aux_logits=True,init_weights=False)
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



def load_model():
    if not CHECKPOINT.is_file(): raise FileNotFoundError(f'Model checkpoint not found: {CHECKPOINT}')
    data=torch.load(CHECKPOINT,map_location='cpu',weights_only=True)
    model=MultiBranchTSNGRU(2); model.load_state_dict(data['model'],strict=True); model.eval()
    return model, data['actions']

def predict(model,actions,path):
    cap=cv2.VideoCapture(str(path))
    try:
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not cap.isOpened() or n<=0: raise ValueError('Could not read the selected video.')
        frames=[]
        for i in range(4):
            lo=min(n-1,i*n//4); hi=min(n-1,max(lo,(i+1)*n//4-1)); idx=(lo+hi)//2
            cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,im=cap.read()
            if not ok: raise ValueError(f'Could not decode frame {idx}.')
            im=cv2.resize(cv2.cvtColor(im,cv2.COLOR_BGR2RGB),(224,224))
            frames.append(torch.from_numpy(((im.astype(np.float32)/255.-MEAN)/STD).transpose(2,0,1).copy()))
    finally: cap.release()
    clip=torch.stack(frames); inputs=torch.stack([clip,clip]).unsqueeze(0)
    with torch.inference_mode(): scores=torch.softmax(model(inputs),dim=1)[0]
    idx=int(scores.argmax()); action_id=int(actions[idx])
    return action_id,ACTIONS.get(action_id,f'Action {action_id}'),float(scores[idx])

def annotate_frame(frame, label):
    """Overlay the clip-level prediction; this is not frame-by-frame detection."""
    frame=frame.copy()
    h,w=frame.shape[:2]
    scale=max(.45, min(w/850, 1.2)); thickness=max(1,round(scale*2))
    lines=[f'Action: {label}']
    line_height=max(20,int(32*scale)); top=min(h, line_height+12)
    overlay=frame.copy(); cv2.rectangle(overlay,(0,0),(w,top),(10,15,25),-1)
    frame=cv2.addWeighted(overlay,.78,frame,.22,0)
    for i,line in enumerate(lines):
        cv2.putText(frame,line,(8, min(h-3, int(24*scale)+i*line_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),thickness,cv2.LINE_AA)
    return frame

def export_video(source, destination, label):
    """Write an annotated MP4 containing the original video frames, without audio."""
    if Path(source).resolve()==Path(destination).resolve():
        raise ValueError('Choose a different file for the output video.')
    cap=cv2.VideoCapture(str(source)); writer=None
    try:
        if not cap.isOpened(): raise ValueError('Could not open the input video.')
        fps=cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps<=0: raise ValueError('Input video has no valid frame rate.')
        ok,frame=cap.read()
        if not ok: raise ValueError('Could not decode the input video.')
        h,w=frame.shape[:2]
        writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        if not writer.isOpened(): raise ValueError('Could not create the MP4 output video.')
        while ok:
            writer.write(annotate_frame(frame,label)); ok,frame=cap.read()
    finally:
        cap.release()
        if writer is not None: writer.release()

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--video'); args=parser.parse_args()
    torch.set_num_threads(min(4,torch.get_num_threads()))
    if args.video:
        model,actions=load_model(); _,label,_=predict(model,actions,args.video)
        print(f'Predicted action: {label}'); return
    import tkinter as tk
    from tkinter import filedialog,ttk
    from PIL import Image,ImageTk
    root=tk.Tk(); root.title('IVP Group 19 | Action recognition'); root.geometry('900x640'); root.minsize(760,580)
    ttk.Label(root,text='Action Recognition Demo',font=('Segoe UI',22,'bold')).pack(pady=(16,3))
    ttk.Label(root,text='Select a video, preview it, then classify the action.').pack()
    chosen=tk.StringVar(value=str(HERE/'sample_video.avi')); status=tk.StringVar(value='Sample selected. Press Classify to begin.')
    preview=ttk.Label(root,text='Video preview',anchor='center'); preview.pack(pady=12)
    ttk.Label(root,textvariable=chosen,wraplength=820).pack(pady=4)
    row=ttk.Frame(root); row.pack(pady=8)
    state={'capture':None,'timer':None,'model':None,'actions':None,'result':None}
    def refresh_preview():
        cap=state['capture']
        if cap is None: return
        ok,frame=cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,frame=cap.read()
        if not ok: preview.config(text='Preview unavailable',image=''); return
        if state['result'] is not None:
            _,label,_=state['result']; frame=annotate_frame(frame,label)
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); im.thumbnail((540,290))
        photo=ImageTk.PhotoImage(im); preview.configure(image=photo,text=''); preview.image=photo
        fps=cap.get(cv2.CAP_PROP_FPS); delay=int(1000/fps) if fps and 1<=fps<=60 else 80
        state['timer']=root.after(max(20,delay),refresh_preview)
    def open_preview(path):
        if state['timer'] is not None: root.after_cancel(state['timer']); state['timer']=None
        if state['capture'] is not None: state['capture'].release()
        cap=cv2.VideoCapture(path)
        if not cap.isOpened(): state['capture']=None; preview.config(text='Unable to preview video',image=''); return
        state['capture']=cap; refresh_preview()
    def browse():
        examples=HERE/'example_videos'
        name=filedialog.askopenfilename(title='Choose a video',initialdir=str(examples if examples.is_dir() else HERE),filetypes=[('Video files','*.avi *.mp4 *.mov *.mkv'),('All files','*.*')])
        if name:
            state['result']=None; save_button.config(state='disabled')
            chosen.set(name); status.set('Video selected. Press Classify.'); open_preview(name)
    browse_button=ttk.Button(row,text='Choose video',command=browse); browse_button.pack(side='left',padx=6)
    def classify():
        state['result']=None; save_button.config(state='disabled')
        button.config(state='disabled'); browse_button.config(state='disabled'); status.set('Loading model and analyzing video...')
        selected=chosen.get()
        def worker():
            try:
                if state['model'] is None: state['model'],state['actions']=load_model()
                aid,label,score=predict(state['model'],state['actions'],selected)
                result=(aid,label,score); msg=f'Predicted action: {label}'
            except Exception as exc: result=None; msg=f'Could not classify video: {exc}'
            def finished():
                state['result']=result; status.set(msg); button.config(state='normal'); browse_button.config(state='normal')
                save_button.config(state='normal' if result is not None else 'disabled')
            try: root.after(0,finished)
            except RuntimeError: pass
        threading.Thread(target=worker,daemon=True).start()
    button=ttk.Button(row,text='Classify',command=classify); button.pack(side='left',padx=6)
    def save_output():
        if state['result'] is None: return
        destination=filedialog.asksaveasfilename(title='Save annotated video',defaultextension='.mp4',filetypes=[('MP4 video','*.mp4')],initialfile='classified_output.mp4')
        if not destination: return
        _,label,_=state['result']; source=chosen.get()
        button.config(state='disabled'); browse_button.config(state='disabled'); save_button.config(state='disabled')
        status.set('Saving annotated output video...')
        def worker():
            try:
                export_video(source,destination,label)
                msg=f'Output saved: {destination}\nVideo contains the clip-level prediction; audio is not included.'
            except Exception as exc: msg=f'Could not save output video: {exc}'
            def finished():
                status.set(msg); button.config(state='normal'); browse_button.config(state='normal'); save_button.config(state='normal')
            try: root.after(0,finished)
            except RuntimeError: pass
        threading.Thread(target=worker,daemon=True).start()
    save_button=ttk.Button(row,text='Save output video',command=save_output,state='disabled'); save_button.pack(side='left',padx=6)
    ttk.Label(root,textvariable=status,font=('Segoe UI',14,'bold'),wraplength=780,justify='center').pack(pady=12)
    ttk.Label(root,text='One predicted action for the whole video',foreground='#555555').pack(side='bottom',pady=(2,8))
    ttk.Label(root,text='Recognized actions: '+', '.join(ACTIONS.values()),wraplength=820,
              justify='center',foreground='#555555').pack(side='bottom',pady=(2,4))
    def close():
        if state['timer'] is not None: root.after_cancel(state['timer'])
        if state['capture'] is not None: state['capture'].release()
        root.destroy()
    root.protocol('WM_DELETE_WINDOW',close); open_preview(chosen.get()); root.mainloop()
if __name__=='__main__': main()
