"""Import only a simulation-selected Matcher, never its trained old head."""
import hashlib
from pathlib import Path
import torch


def load_selected_matcher(adapter, path, expected_sha):
    path=Path(path)
    actual=hashlib.sha256(path.read_bytes()).hexdigest()
    if not expected_sha or actual!=expected_sha:
        raise ValueError('selected Matcher checkpoint hash differs')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved.get('stage')!='matcher' or saved.get('epoch',0)<=0:
        raise ValueError('requires a completed simulation Matcher stage checkpoint')
    state={k[len('matcher.'):]:v for k,v in saved['model'].items() if k.startswith('matcher.')}
    if not state:
        raise ValueError('no explicit Matcher state in selected checkpoint')
    adapter.load_state_dict(state,strict=True)
    adapter.set_frozen(True)
    for k,v in adapter.state_dict().items():
        if not torch.equal(v.cpu(),state[k].cpu()):
            raise AssertionError('Matcher import changed tensor '+k)
    return dict(path=str(path),sha256=actual,epoch=saved['epoch'],
        updates=saved.get('updates'),exposures=saved.get('exposures'),
        old_head_imported=False,optimizer_imported=False,matcher_frozen=True)
