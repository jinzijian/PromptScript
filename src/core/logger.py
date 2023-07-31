from collections import defaultdict
from collections.abc import Iterable
from torch.utils.tensorboard import SummaryWriter
from numbers import Number
from .utils import ntuple
from . import tools, metrics
import json, csv
import torch
import os, sys

def format_number(number, max_length):
    str_num = f"{number:.{max_length}f}"
    if float(str_num) == 0:
        str_num = f"{number:.{max_length}e}"

    return str_num

class TBLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.writer = None
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        self.history = defaultdict(list)
        self.iterator = defaultdict(int)

    def safe(self, write):
        if write:
            self.writer = SummaryWriter(self.log_path)
        else:
            if self.writer is not None:
                self.writer.close()
                self.writer = None
            for name in self.mean:
                self.history[name].append(self.mean[name])
        return

    def reset(self):
        self.tracker = defaultdict(int)
        self.counter = defaultdict(int)
        self.mean = defaultdict(int)
        return

    def append(self, result, tag, n=1, mean=True):
        for k in result:
            name = '{}/{}'.format(tag, k)
            self.tracker[name] = result[k]
            if mean:
                if isinstance(result[k], Number):
                    self.counter[name] += n
                    self.mean[name] = ((self.counter[name] - n) * self.mean[name] + n * result[k]) / self.counter[name]
                elif isinstance(result[k], Iterable):
                    if name not in self.mean:
                        self.counter[name] = [0 for _ in range(len(result[k]))]
                        self.mean[name] = [0 for _ in range(len(result[k]))]
                    _ntuple = ntuple(len(result[k]))
                    n = _ntuple(n)
                    for i in range(len(result[k])):
                        self.counter[name][i] += n[i]
                        self.mean[name][i] = ((self.counter[name][i] - n[i]) * self.mean[name][i] + n[i] *
                                              result[k][i]) / self.counter[name][i]
                else:
                    raise ValueError('Not valid data type')
        return

    def write(self, tag, metric_names):
        names = ['{}/{}'.format(tag, k) for k in metric_names]
        evaluation_info = []
        for name in names:
            tag, k = name.split('/')
            if isinstance(self.mean[name], Number):
                s = self.mean[name]
                evaluation_info.append('{}: {:.4f}'.format(k, s))
                if self.writer is not None:
                    self.iterator[name] += 1
                    self.writer.add_scalar(name, s, self.iterator[name])
            elif isinstance(self.mean[name], Iterable):
                s = tuple(self.mean[name])
                evaluation_info.append('{}: {}'.format(k, s))
                if self.writer is not None:
                    self.iterator[name] += 1
                    self.writer.add_scalar(name, s[0], self.iterator[name])
            else:
                raise ValueError('Not valid data type')
        info_name = '{}/info'.format(tag)
        info = self.tracker[info_name]
        info[2:2] = evaluation_info
        info = '  '.join(info)
        if self.writer is not None:
            self.iterator[info_name] += 1
            self.writer.add_text(info_name, info, self.iterator[info_name])
        return info

    def flush(self):
        self.writer.flush()
        return

# class JsonLogger:
#     def __init__(self, save_path):
#         self.save_path = save_path
#         tools.makedir_exist_ok(self.save_path)

#     def write(self, info_dict):
#         json_data = json.dumps(info_dict)
#         with open(self.save_path, 'a') as file:
#             file.write(json_data + '\n')

class TXTLogger:
    def __init__(self, save_path):
        self.save_path = save_path
        tools.makedir_exist_ok(self.save_path)

    def format(self, info_dict):
        msgs = []
        for k, v in info_dict.items():
            if isinstance(v, metrics.Metric):
                v = v.last
            if isinstance(v, torch.Tensor):
                v = v.item()
            if isinstance(v, float):
                msg = f'{str(k)}={format_number(v, 5)}'
            else:
                msg = f'{str(k)}={v}'
            msgs.append(msg)

        return ','.join(msgs)

    def write(self, info_dict):
        info = self.format(info_dict)
        with open(self.save_path, 'a') as file:
            file.write(info + '\n')

class ModelAsset:
    def __init__(self, save_dir, model, optimizer=None, scheduler=None, ema=None):
        self.save_dir = save_dir
        tools.makedir_exist_ok(self.save_dir)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
    
    def save_model(self, tag='best'):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, f'model_{tag}.pth'))
    
    def save(self, epoch):
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_last.pth'))
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(self.save_dir, 'optimizer.pth'))
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), os.path.join(self.save_dir, 'scheduler.pth'))
        torch.save(epoch, os.path.join(self.save_dir, 'epoch.pth'))
        if self.ema is not None:
            torch.save(self.ema.state_dict(), os.path.join(self.save_dir, 'ema.pth'))

    def resume(self, strict=False, ddp=False):
        # model_state = torch.load(os.path.join(self.save_dir, 'model.pth'))
        # self.model.load_state_dict(model_state)
        self.model = tools.load_weights_from_ckpt(
            self.model, 
            os.path.join(self.save_dir, 'model_last.pth'), 
            strict=strict,
            rename_fn=tools.rename_ddp_state_names if ddp else None)

        optimizer_state = torch.load(os.path.join(self.save_dir, 'optimizer.pth'))
        self.optimizer.load_state_dict(optimizer_state)

        if self.scheduler is not None:
            scheduler_state = torch.load(os.path.join(self.save_dir, 'scheduler.pth'))
            self.scheduler.load_state_dict(scheduler_state)

        if self.ema is not None and os.path.exists(os.path.join(self.save_dir, 'ema.pth')):
            ema_state = torch.load(os.path.join(self.save_dir, 'ema.pth'))
            self.ema.load_state_dict(ema_state)

        self.epoch = torch.load(os.path.join(self.save_dir, 'epoch.pth'))            

        assets = {
            'model': self.model,
            'optimizer': self.optimizer,
            'epoch': self.epoch,
        }
        if self.scheduler is not None:
            assets.update({'scheduler': self.scheduler})
        if self.ema is not None:
            assets.update({'ema': self.ema})
        return assets

def make_logger(type, path):
    if type == 'tb':
        logger = TBLogger(path)
    elif type == 'txt':
        logger = TXTLogger(path)
    elif type == 'json':
        pass
        # logger = JsonLogger(path)
    return logger
