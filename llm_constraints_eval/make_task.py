import os, sys
import pandas as pd
import pickle
import core
import json
import numpy as np
import itertools

def parse_constraint_type(constraint_type):
    cmds = constraint_type.split('-')
    num_samples = int(cmds[-1])
    constraints = {}
    for i in range(len(cmds) - 1):
        c, r = cmds[i].split('=')
        constraints[c] = r
    return num_samples, constraints

def make_constraints(constraint_type, data, seed):
    choices = []
    n, constraints = parse_constraint_type(constraint_type)
    for i in range(len(data)):
        d = data[str(i)]
        d_choices = {'s':[], 'i':[], 'e':[]}
        for c, r in constraints.items():
            for k in range(1, 4): # max 3 constraints per type
                if f'{c}{k}' not in d:
                    continue
                if r == 'h':
                    if 4 <= int(d[f'{c}{k}']) <= 5:
                        d_choices[c].append(k)
                elif r == 'l':
                    if 1 <= int(d[f'{c}{k}']) <= 3:
                        d_choices[c].append(k)
        is_valid = True
        for c in d_choices:
            if len(d_choices[c]) == 0:
                if c in constraints:
                    is_valid = False
                    break
                else:
                    d_choices[c] = [None]
        if is_valid:
            d_combinations = itertools.product([i], d_choices['s'], d_choices['i'], d_choices['e'])
            choices.extend(d_combinations)

    if n == -1:
        return choices
    elif len(choices) < n:
        raise ValueError(f"{constraints}, {len(choices)} found but {n} required.")
    else:
        np.random.seed(seed)
        choices = np.random.permutation(choices)[:n]
        choices = sorted(choices, key=lambda x: x[0])
        return choices
    
if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    
    core.tools.makedir_exist_ok(os.path.join(cfg.save_path))
    
    with open(os.path.join(cfg.data_path), 'r') as f:
        data = json.load(f)
    
    for constraint_type in cfg.constraint.types:
        constraints = make_constraints(constraint_type, data, seed=cfg.constraint.seed)
        df_task = pd.DataFrame(data=constraints, columns=['item_id', 's', 'i', 'e'])
        filename = os.path.join(cfg.save_path, f'task_{constraint_type}.csv')
        with open(filename, 'w') as f:
            df_task.to_csv(f, index=False)
