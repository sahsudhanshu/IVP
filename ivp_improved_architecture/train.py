"""Train development and refit models; optionally evaluate the held-out view once."""
import argparse, csv, json, random, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import ACTIONS, ACTION_NAMES, VideoGroups, inventory
from model import ViewInvariantActionNet, paired_loss

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def now(): return datetime.now(timezone.utc).isoformat(timespec='seconds')
def log(out, record):
    record={'timestamp_utc':now(),**record}; line=json.dumps(record,allow_nan=False)
    print(line,flush=True)
    with (out/'training.jsonl').open('a',encoding='utf-8') as f: f.write(line+'\n')

def metrics(targets,predictions,loss):
    cm=np.zeros((len(ACTIONS),len(ACTIONS)),dtype=np.int64)
    for target,pred in zip(targets,predictions): cm[target,pred]+=1
    tp=cm.diagonal().astype(float); support=cm.sum(axis=1); predicted=cm.sum(axis=0)
    precision=np.divide(tp,predicted,out=np.zeros_like(tp),where=predicted!=0)
    recall=np.divide(tp,support,out=np.zeros_like(tp),where=support!=0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros_like(tp),where=(precision+recall)!=0)
    return {'loss':float(loss),'accuracy':float(tp.sum()/cm.sum()),'macro_precision':float(precision.mean()),
            'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),'support':int(cm.sum())},cm

def evaluate(model,loader,device):
    model.eval(); targets=[]; predictions=[]; losses=0
    with torch.inference_mode():
        for clips,labels in loader:
            b,v,t,c,h,w=clips.shape
            logits,_=model(clips.to(device).reshape(b*v,t,c,h,w))
            # Validation has two source views; test has only the held-out third view.
            probs=logits.reshape(b,v,-1).softmax(-1).mean(1)
            losses+=nn.functional.nll_loss(probs.clamp_min(1e-9).log(),labels.to(device),reduction='sum').item()
            targets.extend(labels.tolist()); predictions.extend(probs.argmax(-1).cpu().tolist())
    result,cm=metrics(targets,predictions,losses/len(targets))
    return result,cm,targets,predictions

def fit(phase,groups,validation,args,out,device,epochs):
    seed_all(args.seed)
    model=ViewInvariantActionNet(frames=args.frames,pretrained=not args.no_pretrained).to(device)
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':args.backbone_lr},
            {'params':list(model.project.parameters())+[model.position]+list(model.temporal.parameters())+
             list(model.attention.parameters())+list(model.norm.parameters())+list(model.classifier.parameters()),'lr':args.lr}],
             weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    scaler=torch.amp.GradScaler(device.type,enabled=device.type=='cuda')
    train_loader=DataLoader(VideoGroups(groups,'train',args.frames,args.size),batch_size=args.batch_size,shuffle=True,
                            num_workers=args.workers,pin_memory=device.type=='cuda')
    val_loader=DataLoader(VideoGroups(validation,'validation',args.frames,args.size),batch_size=args.batch_size,
                          num_workers=args.workers,pin_memory=device.type=='cuda') if validation else None
    best=(-1.,-1.); best_epoch=0
    for epoch in range(1,epochs+1):
        model.train(); count=0; correct=0; total_loss=0.; start=time.time(); lr=optimizer.param_groups[1]['lr']
        for clips,labels in train_loader:
            clips=clips.to(device); labels=labels.to(device); b,v,t,c,h,w=clips.shape
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=device.type=='cuda'):
                logits,features=model(clips.reshape(b*v,t,c,h,w)); logits=logits.reshape(b,v,-1); features=features.reshape(b,v,-1)
                loss,_,_=paired_loss(logits,features,labels,args.consistency_weight)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
            count+=b; total_loss+=loss.item()*b
            correct+=(logits.detach().softmax(-1).mean(1).argmax(-1)==labels).sum().item()
        row={'phase':phase,'epoch':epoch,'train_loss':total_loss/count,'train_accuracy':correct/count,
             'learning_rate':lr,'backbone_learning_rate':optimizer.param_groups[0]['lr'],'seconds':round(time.time()-start,2)}
        if val_loader:
            val,_,_,_=evaluate(model,val_loader,device)
            row.update({'validation_'+k:v for k,v in val.items()})
            key=(val['accuracy'],val['macro_f1'])
            if key>best:
                best=key; best_epoch=epoch
                torch.save({'state_dict':model.state_dict(),'frames':args.frames,'size':args.size,'actions':ACTIONS,
                            'epoch':epoch,'validation':val,'architecture':'ViewInvariantActionNet'},out/'development_best.pt')
        log(out,row); scheduler.step()
    if not validation:
        torch.save({'state_dict':model.state_dict(),'frames':args.frames,'size':args.size,'actions':ACTIONS,
                    'epoch':epochs,'architecture':'ViewInvariantActionNet'},out/'refit.pt')
    return model,best_epoch

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True,help='Extracted dataset directory containing multiview_action_videos')
    p.add_argument('--output',required=True,help='New output directory; must not already exist')
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--frames',type=int,default=8)
    p.add_argument('--size',type=int,default=224); p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--backbone-lr',type=float,default=3e-5)
    p.add_argument('--weight-decay',type=float,default=1e-4); p.add_argument('--consistency-weight',type=float,default=.1)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--workers',type=int,default=0)
    p.add_argument('--no-pretrained',action='store_true',help='Random initialization; intended for smoke tests')
    p.add_argument('--evaluate-test',action='store_true',help='Evaluate target view 3 after refit; do not tune on test')
    p.add_argument('--smoke',action='store_true',help='Run a tiny diagnostic subset; never report as model performance')
    args=p.parse_args()
    if min(args.epochs,args.frames,args.batch_size)<=0 or args.size<32 or args.workers<0 or args.consistency_weight<0: p.error('Invalid numeric argument')
    if args.smoke and args.evaluate_test: p.error('Smoke runs cannot evaluate the target test set')
    groups=inventory(args.data)
    if len(groups)!=418 or {g['action_id'] for g in groups}!=set(ACTIONS):
        raise ValueError(f'Expected the project dataset with 418 complete groups and ten actions, got {len(groups)} groups')
    train=[g for g in groups if g['subject']!=10]; val=[g for g in groups if g['subject']==10]
    if not val or not train: raise ValueError('Source-subject split is empty')
    if args.smoke: train=train[:2]; val=val[:2]; args.epochs=1
    out=Path(args.output); out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    (out/'split_manifest.json').write_text(json.dumps({'development_train':train,'validation':val,
        'refit_train':train+val,'test':groups if args.evaluate_test else [],'actions':ACTIONS},indent=2),encoding='utf-8')
    log(out,{'event':'start','device':str(device),'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
             'pytorch':torch.__version__,'train_groups':len(train),'validation_groups':len(val),
             'pretrained':not args.no_pretrained,'smoke':args.smoke,'config':vars(args)})
    _,best_epoch=fit('development',train,val,args,out,device,args.epochs)
    if args.smoke:
        log(out,{'event':'smoke_complete','warning':'Diagnostic subset only; not an experiment result'}); return
    log(out,{'event':'refit_start','selected_epoch':best_epoch})
    model,_=fit('refit',train+val,None,args,out,device,best_epoch)
    if args.evaluate_test:
        test_loader=DataLoader(VideoGroups(groups,'test',args.frames,args.size),batch_size=args.batch_size,
                               num_workers=args.workers,pin_memory=device.type=='cuda')
        result,cm,ys,ps=evaluate(model,test_loader,device)
        (out/'test_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        with (out/'test_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['true_action_id','predicted_action_id','subject','event','path'])
            for g,y,pred in zip(groups,ys,ps): writer.writerow([ACTIONS[y],ACTIONS[pred],g['subject'],g['event'],g['paths'][2]])
        with (out/'confusion_matrix.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f); writer.writerow(['true/predicted']+ACTION_NAMES)
            for name,row in zip(ACTION_NAMES,cm.tolist()): writer.writerow([name]+row)
        log(out,{'event':'target_test_complete',**result})
    else: log(out,{'event':'complete','test_status':'not evaluated'})
if __name__=='__main__': main()
