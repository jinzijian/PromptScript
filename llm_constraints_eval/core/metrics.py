import torch
import numpy as np
# from collections import Iterable
from collections.abc import Iterable

class Metric():
    def __init__(self, name, order='up'):
        self.name = name
        self.order = order
        self.last = None
        self.best = None
        self.worst = None
        self.improved = False

    def __call__(self, val):
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.last = val
        if self.best is None:
            self.best = val
            self.worst = val
            self.improved = True
        else:
            if self.order == 'up':
                self.improved = self.last > self.best
                self.best = max(self.last, self.best)
                self.worst = min(self.last, self.worst)
            elif self.order == 'down':
                self.improved = self.last < self.best
                self.best = min(self.last, self.best)
                self.worst = max(self.last, self.worst)

class MetricGroup():
    def __init__(self, metrics, pivot_name=None):
        self.metrics = {}
        for m in metrics:
            self.metrics[m.name] = m
        self.pivot_name = pivot_name
        self.pivot = None
        self.improved = False

    def __call__(self, vals):
        for k, v in vals.items():
            if k in self.metrics:
                self.metrics[k](v)
        if self.pivot_name is not None:
            self.pivot = self.metrics[self.pivot_name].best
            self.improved = self.metrics[self.pivot_name].improved
    
    def items(self):
        items = {}
        for k, v in self.metrics.items():
            items[k] = v.last
        return items

class BatchBuffer:
    def __init__(self):
        self.buf = None

    def concat(self, x, buf):
        if isinstance(x, torch.Tensor):
            if buf is None:
                buf = x.detach()
            else:
                buf = torch.cat([buf, x.detach()], dim=0)
        elif isinstance(x, Iterable):
            buf = np.cat([buf, x], dim=0)
        else:
            buf = np.cat([buf, [x]], dim=0)
        return buf

    def __call__(self, x):
        if isinstance(x, dict):
            if self.buf is None:
                self.buf = {}
            for k, v in x.items():
                self.buf[k] = self.concat(v, self.buf.get(k, None))
        else:
            self.buf = self.concat(x, self.buf)
        return self.buf
    
    def reset(self):
        self.buf = None


def recur(fn, input, *args):
    if isinstance(input, torch.Tensor) or isinstance(input, np.ndarray):
        output = fn(input, *args)
    elif isinstance(input, list):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
    elif isinstance(input, tuple):
        output = []
        for i in range(len(input)):
            output.append(recur(fn, input[i], *args))
        output = tuple(output)
    elif isinstance(input, dict):
        output = {}
        for key in input:
            output[key] = recur(fn, input[key], *args)
    elif isinstance(input, str):
        output = input
    elif input is None:
        output = None
    else:
        raise ValueError('Not valid input type')
    return output