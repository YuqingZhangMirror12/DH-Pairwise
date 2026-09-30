import torch
from torch import nn
from torch.nn import functional as F


class PrimalDual(nn.Module):
    def __init__(self, dim=96):
        super().__init__()
        self.primal = nn.Linear(dim, dim, bias=False)
        self.dual = nn.Linear(dim, dim, bias=False)

    def forward(self, a, b):
        # Independent role projections, each shared across fragments.
        pa, pb = (F.normalize(self.primal(x).float(), dim=-1, eps=1e-6) for x in (a, b))
        da, db = (F.normalize(self.dual(x).float(), dim=-1, eps=1e-6) for x in (a, b))
        affinity = .5*(pa@db.transpose(1, 2)+da@pb.transpose(1, 2))
        return affinity, (pa, da, pb, db)
