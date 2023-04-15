import torch
import torch.nn as nn
import torch.functional as F

class MLPHead(nn.Module):
    def __init__(self, dims, act=None, dropout=0.0):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)
            if is_last:
                continue
            if act is None:
                act = nn.ReLU()
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        for l in self.mlp:
            try:
                nn.init.xavier_normal_(l.weight)
            except:
                pass
            
    def forward(self, x):
        if x.ndim > 2:
            x = x.flatten(1)
        return self.mlp(x)