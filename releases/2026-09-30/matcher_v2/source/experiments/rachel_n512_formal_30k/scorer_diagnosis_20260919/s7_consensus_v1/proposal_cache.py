"""Bound deterministic proposals only. Never substitutes for full online Q."""
import hashlib
import json
import os
from pathlib import Path

import torch


class ProposalCache:
    def __init__(self,root,binding):
        self.root=Path(root)
        self.root.mkdir(parents=True,exist_ok=True)
        self.binding=json.loads(json.dumps(binding,sort_keys=True))
        raw=json.dumps(self.binding,sort_keys=True,separators=(',',':')).encode()
        self.signature=hashlib.sha256(raw).hexdigest()
        marker=self.root/'binding.json'
        # Driver initializes on rank0 before the distributed barrier. A second
        # independent caller cannot silently reuse differently bound evidence.
        if marker.exists():
            if json.loads(marker.read_text())!=self.binding:
                raise ValueError('proposal cache binding differs')
        else:
            with marker.open('x') as stream:
                json.dump(self.binding,stream,sort_keys=True,indent=2)
        self.hits=0;self.misses=0

    def get(self,pair_id,builder,pair):
        key=hashlib.sha256(pair_id.encode()).hexdigest()
        path=self.root/key[:2]/(key+'.pt')
        if path.exists():
            record=torch.load(path,map_location='cpu',weights_only=False)
            if record['binding_signature']!=self.signature or record['pair_id']!=pair_id:
                raise ValueError('proposal cache identity mismatch')
            self.hits+=1
            return record['proposals']
        proposals=builder(pair)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
        torch.save(dict(schema='s7-consensus-proposals/1',binding_signature=self.signature,
                        pair_id=pair_id,proposals=proposals),temporary)
        os.replace(temporary,path)
        self.misses+=1
        return proposals
