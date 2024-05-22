import os, sys
from openai import OpenAI
import pandas as pd
import pickle
import core
import json
import numpy as np

def make_client(cfg):
    llm_cfg = cfg[cfg.run.llm_type]
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=llm_cfg.api_key
    )
    if not cfg.run.llm_type in ['gpt3', 'gpt4']:
        client.base_url = llm_cfg.base_url
    
    return client
    
def make_constraints(cfg, content):
    types = cfg.constraints.types
    seed = cfg.constraints.seed
    msg = ''
    for t in range(len(types)):
        key = f'{types[t]}_constraints'
        np.random.seed(seed)
        i = np.random.randint(0, len(content[key]))
        if len(msg) > 0:
            msg = msg + ' '
        msg = msg + f'({t+1}) ' + content[key][i]
    return msg
    
def make_task_constraints(cfg, content):
    TEMPLATE = (
        'Please tell me how to complete the task: <TASK> ' + 
        'There are the following constraints: <CS> Time constraint: <TC> ' + 
        'Please find a way to overcome the constraints and complete the task within the time limit. ' + 
        'If you can complete the task within the time limit, please respond with the number 0 and a brief solution. ' + 
        'If the task cannot be completed, please respond with the number 1 and the reason for the inability to complete. ' + 
        'Please output the answer in the following JSON format: {0, <solution>} or {1, <reason for inability to complete>}.'
    )
    constraints = make_constraints(cfg, content)
    ret = (
        TEMPLATE.replace('<TASK>', content['task'])
                .replace('<TC>', content['time_constraints'][0])
                .replace('<CS>', constraints) 
    )  
    return ret
    
def get_response(cfg, client, content_string):
    llm_cfg = cfg[cfg.run.llm_type]
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": llm_cfg.role,
                "content": content_string,
            }
        ],
        model= llm_cfg.model,
        temperature = 0
    )
    return chat_completion.choices[0].message.content

def save_list_to_file(my_list, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(my_list, file)

def save_responses(cfg, resp_buffer):
    rows = {'prompts': [r[0] for r in resp_buffer], 
            'resps': [r[1] for r in resp_buffer]}
    df = pd.DataFrame.from_dict(rows)
    fname = (
        f'{cfg.run.llm_type}' + 
        '_types=' +
        '-'.join(cfg.constraints.types) +
        f'_seed={cfg.constraints.seed}' +  
        '.json'
    )
    df.to_json(fname, orient='index', indent=4)

if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    
    core.tools.makedir_exist_ok(os.path.join(cfg.resp_save_path, cfg.run.llm_type))
    
    # big_df = pd.read_pickle(os.path.join(cfg.data.path))
    with open(os.path.join(cfg.data.path), 'r') as f:
        data = json.load(f)
        
    client = make_client(cfg)
        
    # set parameters
    START = 0
    # END = 1
    END = len(data.keys())
    
    resp_buffer = []
    for i in range(START, END):
        print(i)
        question = make_task_constraints(cfg, data[f'{i}'])
        result = get_response(cfg, client, question)
        resp_buffer.append((question, result))
        if i % 10 == 0 or i == END - 1:
            save_responses(cfg, resp_buffer)