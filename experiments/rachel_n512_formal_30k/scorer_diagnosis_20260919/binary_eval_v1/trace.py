"""Detached hooks on the actual binary forward; not an extra scoring pass."""
import torch

class MLPTrace:
    def __init__(self,head):
        self.head=head;self.clusters=[];self.current={};self.handles=[]
    def __enter__(self):
        if any(isinstance(m,torch.nn.MultiheadAttention) for m in self.head.modules()):
            raise ValueError('binary diagnostic cannot silently export an Attention head')
        def layer_hook(name,module):
            def record(_module,inputs,output):
                if name in self.current:raise ValueError('layer reused unexpectedly within a cluster')
                value=dict(input=inputs[0].detach().cpu().clone(),output=output.detach().cpu().clone(),
                           kind='linear' if isinstance(module,torch.nn.Linear) else 'gelu')
                if value['kind']=='linear':
                    value.update(weight=module.weight.detach().cpu().clone(),bias=module.bias.detach().cpu().clone())
                self.current[name]=value
            return record
        for name,module in self.head.named_modules():
            if name.startswith(('edge_mlp.','cluster_mlp.')) and isinstance(module,(torch.nn.Linear,torch.nn.GELU)):
                self.handles.append(module.register_forward_hook(layer_hook(name,module)))
        def finish(_module,_inputs,_output):
            if not self.current:raise ValueError('binary forward had no MLP layers')
            self.clusters.append(self.current);self.current={}
        self.handles.append(self.head.register_forward_hook(finish));return self
    def __exit__(self,*_):
        for handle in self.handles:handle.remove()
        self.handles=[]

