import os, sys
import re
import openai
import time
import core
import pandas as pd
import numpy as np
import ast

def convert_to_list_of_str(s):
    s = s.strip('[]')
    s = s.split("', '")
    s = [ss.strip("'") for ss in s]
    return s

def load_items(cfg):
    df = pd.read_csv(cfg.data.item_file)
    df['env_constrains'] = df['env_constrains'].apply(convert_to_list_of_str)

    # preprocess topics to similar format as wikihow
    idx = df['source'] != 'wikihow'
    df.loc[idx, 'topic'] = df[idx]['topic'].map(lambda t: f'How to {t}?')

    # preprocess constrains without capitalizating Without
    df['env_constrains'] = df['env_constrains'].apply(lambda c: [s.replace('Without', 'without') for s in c])

    return df

def strip_string(s):
    pattern = r"\d+\."
    s = re.split(pattern, s.strip())
    if len(s) > 1:
        return s[1].strip()
    else:
        return s[0].strip()
    
def load_prompt_template(cfg):
    templates = core.config.get_config(cfg.prompt.template_file)
    template = templates[cfg.prompt.type][f'TEMPLATE_{cfg.prompt.choice}']
    if cfg.prompt.n_constrain > 0:
        template['CONSTRAIN'] = templates['CONSTRAIN'][f'TEMPLATE_{cfg.prompt.constrain_choice}']
    return template

def make_constrains_prompt(template, item, n=1, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(len(item['constrains']))[:n]
    constrains = np.array(item['constrains'])[idx]
    prompt = f"{template['CONSTRAIN']['PREFIX']} "
    for i, c in enumerate(constrains):
        c = c.split(', ')
        condition, step = c[0], c[1]
        pc = f"{template['CONSTRAIN']['ITEM']}".replace('<NUM>', str(i+1))
        pc = pc.replace('<CONDITION>', condition)
        pc = pc.replace('<STEP>', step)
        prompt = prompt + pc
        if i < len(constrains) - 1:
            prompt = prompt + '; '
    return prompt

def make_plain_prompt(cfg, template, item):
    query = f"{template['QUERY']}".replace('<TOPIC>', item['topic'])
    if cfg.prompt.n_constrain > 0:
        c_prompt = make_constrains_prompt(template, item, cfg.prompt.n_constrain, cfg.prompt.seed)
        return f"{query} {c_prompt} {template['OUTPUT_FORMAT']}"
    return f"{query} {template['OUTPUT_FORMAT']}"

def make_cot_prompt(cfg, template, item, response=None):
    if response is None: # initial prompt
        prefix = f"{template['PREFIX']}".replace('<TOPIC>', item['topic'])
        query = f"{template['FIRST_QUERY']}".replace('<TOPIC>', item['topic'])
        if cfg.prompt.n_constrain > 0:
            c_prompt = make_constrains_prompt(template, item, cfg.prompt.n_constrain, cfg.prompt.seed)
            return f"{prefix} {c_prompt} {query} {template['OUTPUT_FORMAT']}"
        return f"{template['PREFIX']} {query} {template['OUTPUT_FORMAT']}"
    else:
        query = f"{template['QUERY']}".replace('<TOPIC>', item['topic'])
        query = query.replace('<PREV_STEP>', response)
        return f"{query} {template['OUTPUT_FORMAT']}"

def parse_plain_respose(resp):
    resp = resp.choices[0].message.content.strip()
    pattern = r"\d+\. "
    resp = re.split(pattern, resp)[1:]
    resp = [r.strip() for r in resp]
    return resp

def parse_cot_respose(resp):
    # TODO
    raise NotImplementedError

def query_gpt(cfg, prompt):
    response = None
    while response is None:
        try:
            response = openai.ChatCompletion.create(model=cfg.run.openai.engine, 
                                                    messages=[{'role': cfg.run.openai.role, 
                                                               'content': prompt}])
        except:
            time.sleep(cfg.run.wait_duration)
            pass
    return response

def main(cfg):
    core.tools.makedir_exist_ok(cfg.run.save_dir)
    item_df = load_items(cfg)
    template = load_prompt_template(cfg)

    # TODO: multiple item inference
    item = {'topic': item_df['topic'].values[1], 
            'constrains': item_df['env_constrains'].values[1]}

    if cfg.prompt.type == 'PLAIN':
        prompt = make_plain_prompt(cfg, template, item)
        resp = query_gpt(cfg, prompt)
        res = parse_plain_respose(resp)
    elif cfg.prompt.type == 'COT':
        prompt = make_cot_prompt(cfg, template, item)
        # TODO: cot iterative query
        raise NotImplementedError

    print(f'PROMPT: {prompt}')
    print(f'RESPONSE: {res}')

if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    openai.api_key = core.config.get_config(cfg.run.credential_file).openai.api_key
    main(cfg)