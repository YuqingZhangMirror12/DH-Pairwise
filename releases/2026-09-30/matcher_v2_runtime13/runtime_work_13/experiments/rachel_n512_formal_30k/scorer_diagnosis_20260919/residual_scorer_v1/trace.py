"""Same-forward hooks, including the actual skip addition; no second pass."""
import torch

from .head import ResidualClusterHead, ResidualBlock


LAYER_KINDS = {
    'edge_mlp.0': 'linear', 'edge_mlp.1': 'gelu',
    'edge_mlp.2': 'linear', 'edge_mlp.3': 'gelu',
    'cluster_mlp.0': 'linear', 'cluster_mlp.1': 'gelu',
    'cluster_mlp.2.down': 'linear', 'cluster_mlp.2.activation': 'gelu',
    'cluster_mlp.2.up': 'linear', 'cluster_mlp.2': 'residual_add',
    'cluster_mlp.3': 'linear',
}


def detached(value):
    return value.detach().cpu().clone()


class ResidualTrace:
    def __init__(self, head):
        self.head = head
        self.clusters = []
        self.current = {}
        self.handles = []

    def __enter__(self):
        if self.handles or self.current or self.clusters:
            raise ValueError('use one fresh trace context per pair')
        if type(self.head) is not ResidualClusterHead:
            raise ValueError('residual diagnostic needs the explicit residual architecture')
        modules = dict(self.head.named_modules())
        actual = {n for n, m in modules.items()
                  if isinstance(m, (torch.nn.Linear, torch.nn.GELU, ResidualBlock))}
        if actual != set(LAYER_KINDS):
            raise ValueError('residual architecture has unexpected or missing layers')
        if any(isinstance(m, (torch.nn.MultiheadAttention, torch.nn.modules.batchnorm._BatchNorm))
               for m in modules.values()):
            raise ValueError('Attention/BatchNorm are not this registered experiment')

        def hook(name, module):
            def record(_module, inputs, output):
                if name in self.current:
                    raise ValueError('layer repeated inside a cluster')
                value = dict(kind=LAYER_KINDS[name], input=detached(inputs[0]), output=detached(output))
                if value['kind'] == 'linear':
                    value.update(weight=detached(module.weight), bias=detached(module.bias))
                elif value['kind'] == 'residual_add':
                    # Recorded from this very forward's up projection. Do not
                    # re-run F(h), even under no_grad, to manufacture a trace.
                    value['branch_output'] = self.current[name + '.up']['output'].clone()
                self.current[name] = value
            return record

        def finish(_module, _inputs, _output):
            if list(self.current) != list(LAYER_KINDS):
                raise ValueError('residual forward did not execute the declared graph once')
            self.clusters.append(self.current)
            self.current = {}

        try:
            for name in LAYER_KINDS:
                self.handles.append(modules[name].register_forward_hook(hook(name, modules[name])))
            self.handles.append(self.head.register_forward_hook(finish))
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()
        self.handles = []
