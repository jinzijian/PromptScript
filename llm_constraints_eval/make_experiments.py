import os
import itertools
import argparse
import copy
from main import *

def argbooltype(v):
    if isinstance(v, bool):
        return v
    elif isinstance(v, int):
        return v > 0
    elif v.lower() in ('yes', 'y', 't', '1', 'true'):
        return True
    elif v.lower() in ('no', 'n', 'f', '0', 'false'):
        return False

parser = argparse.ArgumentParser(description='args')
parser.add_argument('--cfg_file', default='./configs/config_llama3.yml', type=str)
parser.add_argument('--local_models_only', default=True, type=argbooltype)
parser.add_argument('--first_k', default=None, type=int)
args = parser.parse_args()


def make_experiments(args, base_cfg):
    base_cfg['first_k'] = args.first_k
    api_configs = base_cfg['api_configs'] 

    # Define LLMs, templates, and num_constraints
    llms = list(api_configs.keys())
    if args.local_models_only:
        llms = []
        for k in api_configs.keys():
            if k == 'cfg_file':
                continue
            if api_configs[k]['provider'] == 'ollama':
                llms.append(k)
    templates = ['./templates/base.txt', './templates/softmoe.txt']
    # num_constraints = [1, 2, 3]
    num_constraints = [1]
    
    combinations = list(itertools.product(llms, templates, num_constraints))
    cfgs = []
    for comb in combinations:
        cfg_ = copy.deepcopy(base_cfg)
        cfg_['llm'] = comb[0]
        cfg_['query_template_file'] = comb[1]
        cfg_['num_constraints'] = comb[2]
        cfgs.append(cfg_)    
    return cfgs


if __name__ == "__main__":
    base_cfg = core.config.get_config(args.cfg_file)
    api_configs = core.config.get_config(base_cfg.api_configs_file)
    base_cfg['api_configs'] = api_configs

    cfgs = make_experiments(args, base_cfg)
    for cfg in cfgs:
        main(cfg)
