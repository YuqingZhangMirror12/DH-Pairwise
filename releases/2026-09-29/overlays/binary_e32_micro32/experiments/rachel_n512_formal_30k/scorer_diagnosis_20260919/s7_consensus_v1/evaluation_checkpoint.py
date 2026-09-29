"""Immutable validation-epoch weights; no TEST or retrospective epoch search."""
import hashlib
import os
from pathlib import Path

import torch


def metric_identity(metrics):
    """Exclude only registered wall-clock fields, not mathematical results."""
    result={k:v for k,v in metrics.items() if k!='elapsed_seconds'}
    if isinstance(result.get('real_development'),dict):
        result['real_development']={k:v for k,v in result['real_development'].items()
                                    if k!='elapsed_seconds'}
    return result


def _identity(record):
    keys=('binding','stage','epoch','threshold','updates','exposures')
    result={k:record[k] for k in keys}
    result['metrics']=metric_identity(record['metrics'])
    return result


def save_evaluation_checkpoint(path, record):
    """Preserve a validation state; identical crash replay may reuse it.

    A different model, protocol, epoch or threshold can never overwrite it.
    Timing differences alone are not part of the mathematical identity.
    """
    path=Path(path)
    if record.get('stage') not in ('matcher','scorer') or int(record.get('epoch',-1))<0:
        raise ValueError('validation stage and epoch required')
    if path.exists():
        existing=torch.load(path,map_location='cpu',weights_only=False)
        if (_identity(existing)!=_identity(record) or existing['model'].keys()!=record['model'].keys()
                or any(not torch.equal(existing['model'][k],record['model'][k].detach().cpu())
                       for k in existing['model'])):
            raise ValueError('immutable validation checkpoint identity differs')
        return dict(already_existed=True,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    try:
        with temporary.open('xb') as stream:
            torch.save(record,stream)
        os.link(temporary,path)  # Atomic, refuses replacing an existing inode.
    finally:
        if temporary.exists():
            temporary.unlink()
    return dict(already_existed=False,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
