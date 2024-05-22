import os, sys
from openai import OpenAI
import pandas as pd
import pickle
import core
import json

EVAL_PROMPT= \
    'You are given 2 answers for a task on planning under constraints. The task is "<T>".\n' + \
    'The first answer is "<A1>". \n' + \
    'The second answer is "<A2>". \n' + \
    'Please assign a score (0-100 points) the first answer, by comparing it to the second answer. ' + \
        'Assume the second answer is a perfect answer with 100 points. ' + \
    'The scoring criteria has 5 items. ' +\
        'Item 1-4 applies to all answers. For item 5, 5(a) applies if the answer is a solution (answer with selection 0); or 5(b) applies if the answer gives a reason about giving-up (answer with selection 1).' + \
    'The criteria are detailed as follows:\n' + \
    '1. Relevance (20 points):\n' + \
    '0-4 points: The response is unrelated to the question or is an error message.\n' + \
    '5-8 points: The response addresses the topic but misses key aspects or goes off-topic.\n' + \
    '9-12 points: The response is generally relevant but lacks specific details or focus.\n' + \
    '13-16 points: The response is relevant with minor irrelevant parts.\n' + \
    '17-20 points: The response is completely relevant and focused on the question.\n' + \
    '2. Accuracy (20 points):\n' + \
    '0-4 points: The response is largely inaccurate or based on incorrect assumptions.\n' + \
    '5-8 points: The response contains several inaccuracies or dubious judgments.\n' + \
    '9-12 points: The response is more accurate than inaccurate but has noticeable errors.\n' + \
    '13-16 points: The response is mostly accurate with minor errors.\n' + \
    '17-20 points: The response is completely accurate and shows correct judgment.\n' + \
    '3. Thoroughness (20 points):\n' + \
    '0-4 points: The response overlooks most of the given constraints or aspects.\n' + \
    '5-8 points: The response considers some constraints but misses several key aspects.\n' + \
    '9-12 points: The response is moderately thorough but lacks depth in certain areas.\n' + \
    '13-16 points: The response is thorough with only minor aspects overlooked.\n' + \
    '17-20 points: The response thoroughly and comprehensively addresses all aspects and constraints.\n' + \
    '4. Clarity and Conciseness (20 points):\n' + \
    '0-4 points: The response is very unclear, confusing, and overly verbose.\n' + \
    '5-8 points: The response has clarity issues and could be more concise.\n' + \
    '9-12 points: The response is clear but could be expressed more concisely or directly.\n' + \
    '13-16 points: The response is clear and mostly concise with minor improvements needed.\n' + \
    '17-20 points: The response is exceptionally clear, direct, and concisely conveys all necessary information.\n' + \
    'Select only one from 5(a) and 5(b) to score. Use 5(a) if the first answer is 0, and use 5(b) if the first answer is 1.\n' + \
    '5(a) Quality of solution (20 points):\n' + \
    '0-4 points: The solution requires prohibitively high cost in terms of time, effort or money.\n' + \
    '5-8 points: The solution requires significant amount of time, effort or money.\n' + \
    '9-12 points: The solution needs adequate but not too high amount of time, effort or money.\n' + \
    '13-16 points: The solution just needs some common amount of time, effort or money.\n' + \
    '17-20 points: The solution is very cheap and straitforward to execute, requiring only little time, effort or money.\n' + \
    '5(b) Quality of the reason (20 points):\n' + \
    '0-4 points: The reason is completely irrational or infeasible.\n' + \
    '5-8 points: The reason has significant rationality or feasibility issues.\n' + \
    '9-12 points: The reason is somewhat rational and feasible but with questionable elements.\n' + \
    '13-16 points: The reason is rational and feasible with minor impractical aspects.\n' + \
    '17-20 points: The reason is highly rational, comprehensive, and entirely logical.\n' + \
    'Output scores of the first answer in all 5 aspects in a list. Only scores, no other outputs.' 
    

def make_client(cfg):
    llm_cfg = cfg['gpt4']
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=llm_cfg.api_key
    )
    
    return client
    

# def make_content(cfg, resp_string):
#     eval_prompt = 'Please evaluate the following response to a given question. The question is "<Q>". ' + \
#                   'And the response is "<R>". ' + \
#                   'Please evalute based on the following criteria: \n' + \
#                   '1) Does it answer the question? (0-1 point) If none or irrelevant response, give 0 point. Otherwise, 1 point. \n' + \
#                   '2) Is the judgement (0 or 1) given in the response correct or wrong? (0-1 point) \n' + \
#                   '3) Does the response consider all constraints included in the question? (0-2 point) ' + \
#                       'If no constraint is considered, give 0 point. If part of constraints are considered, give 1 point. If all considered, give 2 points. \n' + \
#                   '4) If judgement in the response is 0, is the given reason logically sound? Or if the judgement is 1, is the given solution feasible? (0-3 point) ' + \
#                       'If no reason or solution, give 0 point. If reason or solution has major issue, give 1 point. If reason or solution has minor issue, give 2 point. If no issue, give 3 point. \n' + \
#                   '5) If you are asked the same question, will you give similar answer to the given response? (0-3 point) ' + \
#                       'If different judgement (0 or 1), give 0 point. If same judgement but very different reason or solution, give 1 point. If same judgement but slightly different reason or solution, give 2 point. If same judgement and similar reason or solution, give 3 point. \n' +\
#                   'Output scores of all 5 criteria in this format: {1:score, 2:score, 3:score, 4:score, 5:score}. No other explanations.' 
    
#     prompt = resp_string['prompt']
#     resp = resp_string['resp']
    
#     content = eval_prompt.replace('<Q>', prompt).replace('<R>', resp)
#     return content

def make_content(cfg, prompt, resp, ref_resp, eval_prompt=EVAL_PROMPT):
    content = eval_prompt.replace('<T>', prompt)
    content = content.replace('<A1>', resp)
    content = content.replace('<A2>', ref_resp)
    return content

def get_response(cfg, client, content_string):
    llm_cfg = cfg['gpt4']
    
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
            'resps': [r[1] for r in resp_buffer],
            'ref_resps': [r[2] for r in resp_buffer],
            'scores': [r[3] for r in resp_buffer]}
    df = pd.DataFrame.from_dict(rows)
    fname = (
        'eval=' +
        f'{cfg.run.llm_type}-' +
        f'{cfg.run.ref_llm_type}' + 
        '_types=' +
        '-'.join(cfg.constraints.types) +
        f'_seed={cfg.constraints.seed}' +  
        '.json'
    )
    df.to_json(f'./results/' + fname, orient='index', indent=4)

if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    
    with open(cfg.resp_path, 'r') as f:
        all_resps = json.load(f)
    
    with open(cfg.ref_resp_path, 'r') as f:
        all_ref_resps = json.load(f)
        
    client = make_client(cfg)
    
    all_resp_ids = sorted([int(r) for r in list(all_resps.keys())])
    
    # core.tools.makedir_exist_ok(cfg.run.save_path)
    # set parameters
    START = 0
    # END=2
    END = len(all_resp_ids)
    
    all_evals = []
    for i in range(START, END):
        print(i)
        prompt_ = all_resps[str(all_resp_ids[i])]['prompts']
        resp_ = all_resps[str(all_resp_ids[i])]['resps']
        ref_resp_ = all_ref_resps[str(all_resp_ids[i])]['resps']
        content = make_content(cfg, prompt_, resp_, ref_resp_)
        # print(content)
        result = get_response(cfg, client, content)
        all_evals.append((prompt_, resp_, ref_resp_, result))
        if i % 10 == 0 or i == END - 1:
            save_responses(cfg, all_evals)