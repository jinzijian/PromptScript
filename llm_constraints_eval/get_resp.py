import os, sys
from openai import OpenAI
import pandas as pd
import pickle
import core
import json
import numpy as np

BASIC_TEMPLATE = (
        'Please tell me how to complete the task: <TASK> ' + 
        'There are the following constraints: <CS> Time constraint: <TC> ' + 
        'Please find a way to overcome the constraints and complete the task within the time limit. ' + 
        'If you can complete the task within the time limit, please respond with the number 0 and a brief solution. ' + 
        'If the task cannot be completed, please respond with the number 1 and the reason for the inability to complete. ' + 
        'Please output the answer in the following JSON format: {0, <solution>} or {1, <reason for inability to complete>}.'
    )

ANALYSIS_TEMPLATE =  (
    'Please analyze how to complete the following task: <TASK> ' +
    'The following constraints exist: <CS> Time constraint: <TC> ' +
    'Please consider the following analysis methods: \n' +
    '1. Cost-Benefit Analysis: Compare the total cost of completing the task (including time, resources, opportunity cost) with the expected benefits. Calculate the Return on Investment (ROI). \n' +
    '2. Feasibility Analysis: Assess technical feasibility (availability of necessary technology and tools), economic feasibility (sufficient budget), legal feasibility (compliance with relevant regulations), operational feasibility (adequate manpower and process support). \n' +
    '3. Risk Assessment: Identify potential risks, evaluate the probability and impact of each risk, calculate risk value (probability x impact), propose risk mitigation strategies. \n' + 
    '4. Resource Allocation Analysis: List required resources (manpower, equipment, funds, etc.), evaluate existing resources, develop resource acquisition plans, create resource allocation schedules. \n' +
    '5. Time Management Analysis: Break down the task into specific steps, estimate time needed for each step, create a Gantt chart or critical path diagram, identify time bottlenecks and optimization opportunities. \n' +
    'Please select the most appropriate analysis method for evaluating this task and briefly explain your choice. \n' + 
    'Then, follow the specific steps of the chosen method to analyze and assess whether the task can be completed within the given constraints. \n' +
    'If the assessment shows that the task can be completed, please start with the number 0, provide the chosen analysis method and a brief completion plan based on the analysis. \n' +
    'If the assessment shows that the task cannot be completed, please start with the number 1, provide the chosen analysis method and the specific reasons for inability to complete based on the analysis. \n' +
    'Please output the answer in the following JSON format: \n' +
    '{0, "analysis_method": "<chosen method>", "completion_plan": "<brief completion plan based on analysis>"} ' +
    'or ' +
    '{1, "analysis_method": "<chosen method>", "reason": "<specific reasons for inability to complete based on analysis>"} ' +
    'Stick to the format. Do not output any other things!' 
)


def make_client(cfg):
    llm_cfg = cfg[cfg.llm_type]
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=llm_cfg.api_key
    )
    if not cfg.llm_type in ['gpt3', 'gpt4']:
        client.base_url = llm_cfg.base_url
    
    return client
    
def make_prompt(cfg, data, task):
    TEMPLATE = eval(cfg.template.upper())
    id = str(task['item_id'])
    constraint_keys = {'s': 'skill_constraints', 
                       'i': 'item_constraints', 
                       'e': 'environment_constraints'}
    constraint_prompt = []
    cids, rates = [], []
    for c in ['s', 'i', 'e']:
        if task[c] is None:
            continue
        cp = data[id][constraint_keys[c]][int(task[c]) - 1]
        cr = data[id][f'{c}{task[c]}']
        constraint_prompt.append(cp)
        cids.append(f'{c}{task[c]}')
        rates.append(cr)
    topic_prompt = data[id]['task']
    constraint_prompt = ' '.join(constraint_prompt)
    time_budget_prompt = data[id]['time_constraints'][0]
    ret = (
        TEMPLATE.replace('<TASK>', topic_prompt)
                .replace('<TC>', time_budget_prompt)
                .replace('<CS>', constraint_prompt) 
    )
    return ret, [cids, rates]
    
def get_response(cfg, client, content_string):
    llm_cfg = cfg[cfg.llm_type]
    
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

def save_responses(cfg, fpath, resp_buffer):
    rows = {'prompts': [r[0] for r in resp_buffer], 
            'resps': [r[1] for r in resp_buffer],
            'constraint_id': [r[2][0] for r in resp_buffer],
            'constraint_rate': [r[2][1] for r in resp_buffer]}
    df = pd.DataFrame.from_dict(rows)
    df.to_json(fpath, orient='index', indent=4)

if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    
    api_key = core.config.get_config('./configs/api_key.yml')
    cfg[cfg.llm_type].api_key = api_key[cfg.llm_type]

    core.tools.makedir_exist_ok(os.path.join(cfg.save_path))
    
    # big_df = pd.read_pickle(os.path.join(cfg.data.path))
    with open(os.path.join(cfg.data_path), 'r') as f:
        data = json.load(f)
        
    client = make_client(cfg)
    
    for constraint_type in cfg.constraint.types:
        task = pd.read_csv(os.path.join(cfg.save_path, f'task_{constraint_type}.csv'))
        core.tools.makedir_exist_ok(os.path.join(cfg.save_path, cfg.llm_type))
        # set parameters
        START = 0
        # END = 1
        END = len(task)
        resp_buffer = []
        print(f'Running the {constraint_type} ...')
        for i in range(START, END):
            print(i)
            prompt, rates = make_prompt(cfg, data, task.iloc[i])
            # print(prompt, rates)
            result = get_response(cfg, client, prompt)
            resp_buffer.append((prompt, result, rates))
            if i % 10 == 0 or i == END - 1:
                fname = (
                    f'{cfg.llm_type}_' + 
                    constraint_type +
                    f'_seed={cfg.constraint.seed}' +  
                    '.json'
                )
                fpath = os.path.join(cfg.save_path, cfg.llm_type, cfg.template, fname)
                save_responses(cfg, fpath, resp_buffer)
                # raise ValueError