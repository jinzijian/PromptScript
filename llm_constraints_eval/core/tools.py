import os, sys
import joblib
import json
import glob
import torch
import random
import numpy as np
# from torchvision.utils import save_image
# from collections import Sequence
from collections.abc import Sequence
from . import config

def check_exists(path):
    return os.path.exists(path)


# def makedir_exist_ok(fd):
#     if not os.path.exists(fd):
#         os.makedirs(fd)

def makedir_exist_ok(path):
    is_file = os.path.splitext(path)[1] != ''
    if is_file:
        path = os.path.dirname(path)
    if not os.path.exists(path):
        os.makedirs(path)

def save(input, path, mode='torch'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        joblib.dump(input, path)
    elif mode == 'json':
        with open(path, 'w') as f:
            json.dump(input, f, indent = 2)
    else:
        raise ValueError('Not valid save mode')
    return


# def save_img(img, path, nrow=10, padding=2, pad_value=0, range=None):
#     makedir_exist_ok(os.path.dirname(path))
#     normalize = False if range is None else True
#     save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, range=range)
#     return


def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return joblib.load(path)
    else:
        raise ValueError('Not valid save mode')
    return


def find_latest(root, tar):
    founds = glob.glob(os.path.join(root, f'*{tar}*'))
    assert len(founds) > 0, f'No subdir or file with patter={tar} found in {root}'
    return sorted(founds)[-1]


def rename_ddp_state_names(states):
    renamed_states = {}
    for k, v in states.items():
        if k.split('.')[0] == 'module':
            _name = '.'.join(k.split('.')[1:])
            renamed_states[_name] = v
        else:
            renamed_states[k] = v
    return renamed_states

def load_weights_from_ckpt(model, ckpt_path, strict=False, device='cpu', rename_fn=None):
    states = torch.load(ckpt_path, map_location=device)
    if rename_fn is not None:
        states = rename_fn(states)
    model.load_state_dict(states, strict=strict)
    return model


def load_model_from_ckpt(model, ckpt_path):
    model.load(ckpt_path)
    return model


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def get_port():
    """
    find a free port to used for distributed learning
    """
    pscmd = "netstat -ntl |grep -v Active| grep -v Proto|awk '{print $4}'|awk -F: '{print $NF}'"
    procs = os.popen(pscmd).read()
    procarr = procs.split("\n")
    tt= random.randint(15000, 30000)
    if tt not in procarr:
        return tt
    else:
        return get_port()

def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def apply_fn(module, fn):
    for n, m in module.named_children():
        if hasattr(m, fn):
            exec('m.{0}()'.format(fn))
        if sum(1 for _ in m.named_children()) != 0:
            exec('apply_fn(m,\'{0}\')'.format(fn))
    return

def get_device_ids(devices):
    # if devices is a string, convert it to a list
    if isinstance(devices, str):
        devices = [devices]

    device_ids = []
    for device in devices:
        # if device is already an int, just append it
        if isinstance(device, int):
            device_ids.append(device)
        else:
            # if device is a string, extract the device id
            device = device.replace('cuda:', '').strip()  # remove 'cuda:' and whitespace
            device_id = int(device)  # convert to int
            device_ids.append(device_id)

    return device_ids

def parse_device(device):
    if isinstance(device, Sequence) and not isinstance(device, str):
        device = device[0]
    
    if isinstance(device, str) and 'cuda' in device:
        return device
    else:
        return f'cuda:{device}'
    
def config_distributed(cfg, rank):
    server_name = cfg.get('server_name', 'default')
    server_cfg = config.get_config(cfg.run.server_config_file)
    return server_cfg[server_name]
