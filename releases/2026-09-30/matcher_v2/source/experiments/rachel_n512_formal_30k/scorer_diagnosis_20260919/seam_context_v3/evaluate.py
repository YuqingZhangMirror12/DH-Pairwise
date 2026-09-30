"""Frozen checkpoint evaluation on prepared RAW-mask manifests, no refitting."""
import argparse
from pathlib import Path
import torch
from .config import Config
from .model import SeamContextModel
from .prepare import read,save,sha
from .validate import evaluate_view,metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True)
    p.add_argument('--selection',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--microbatch',type=int,default=4);a=p.parse_args()
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False);selection=read(a.selection)
    if cp['stage']!='B' or cp['selection_state']['best'] is None:raise ValueError('requires selected joint model')
    if selection['binding']!=cp['binding'] or selection['epoch']!=cp['selection_state']['best']['epoch']:
        raise ValueError('selection/checkpoint differ')
    cfg=Config();model=SeamContextModel(cfg).to(a.device);model.load_state_dict(cp['model']);model.eval()
    rows=evaluate_view(model,Path(a.manifest),'B',torch.device(a.device),a.microbatch)
    save(a.out,dict(checkpoint_sha256=sha(a.checkpoint),threshold=selection['threshold'],
        protocol='frozen single checkpoint and SIM CAL threshold; no real refitting',
        metrics=metrics(rows,selection['threshold']),fixed03=metrics(rows,.3),rows=rows))
