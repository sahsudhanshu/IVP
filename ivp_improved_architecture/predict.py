"""Predict one clip with a checkpoint produced by train.py."""
import argparse, torch
from data import ACTIONS, ACTION_NAMES, read_clip
from model import ViewInvariantActionNet

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--video',required=True);args=p.parse_args()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if ckpt.get('actions') != ACTIONS: raise ValueError('Checkpoint action mapping does not match this dataset')
    model=ViewInvariantActionNet(frames=ckpt['frames'],pretrained=False)
    model.load_state_dict(ckpt['state_dict'],strict=True);model.eval()
    clip=read_clip(args.video,ckpt['frames'],ckpt['size'],False).unsqueeze(0)
    with torch.inference_mode(): logits,_=model(clip)
    print('Predicted action:',ACTION_NAMES[int(logits.argmax(-1))])
if __name__=='__main__':main()
