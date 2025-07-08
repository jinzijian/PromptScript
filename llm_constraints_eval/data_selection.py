import os, sys
import numpy as np
import pandas as pd
import core
import json
from task_sampler import *
from query import *
from evaluation import parse_response
import ollama, random, concurrent.futures

"""
    Samples Generation
"""
def one_sample(model_name, prompt, temperature, top_p, seed):
    return ollama.generate(
        model=model_name,
        prompt=prompt,
        stream=False,
        keep_alive='30m',
        options={
            'temperature': temperature,
            'top_p': top_p,
            'seed': seed
        }
    )

def sample_k(model_name, prompt, k=5, temperature=0.8, top_p=0.95):
    with concurrent.futures.ThreadPoolExecutor() as pool:
        futures = [
            pool.submit(one_sample,
                        model_name, prompt, temperature, top_p, 
                        random.randint(1, 2**31 - 1))
            for _ in range(k)
        ]
    return [f.result().response for f in futures]

class ConcurrentLLMQueryHandler(LLMQueryHandler):
    def __init__(self, model_name, api_key, model_provider, save_dir, output_file_prefix='resps',
                 k_samples=5, temperature=0.8, top_p=0.95):
        super().__init__(model_name, api_key, model_provider, save_dir, output_file_prefix)
        self.k_samples = k_samples
        self.temperature = temperature
        self.top_p = top_p

    def query_llm(self, prompt):
        try:
            response = sample_k(self.model_name, prompt, k=self.k_samples, temperature=self.temperature, top_p=self.top_p)
            return response
        except Exception as e:
            print(f"Error querying LLM: {e}")
            return None

"""
    Correctness Utility Functions
"""
def parse_binary_decision(resp):
    decision, _, _ = parse_response(resp)
    if decision is None:
        return -1
    return decision

def verify_by_thresholded_difficulty(resps, difficulty, threshold=4):
    giveup = np.array([parse_binary_decision(r) for r in resps])
    return giveup == (difficulty >= threshold)

"""
    Data Selection Functions
"""
def compute_acc_metric(verifier, responses, **verifier_kwargs):
    total = len(responses)
    all_acc = []
    for i in range(total):
        task_id = responses[i]['task_id']
        constraints = [(c, s) for c, s in responses[i]['constraints'].items()]
        resps = responses[i]['response']
        # assuming single constraint
        scores = verifier(resps, constraints[0][1], **verifier_kwargs)
        acc_at_k = np.mean(scores)
        acc_var = np.var(scores)
        all_acc.append((task_id, constraints[0][0], acc_at_k, acc_var))
    return all_acc

if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    cfg['api_configs'] = core.config.get_config(cfg.api_cfg_file)
    
    # --- Generate samples --- 
    # Load tasks from JSON file
    with open(cfg.data_file, 'r') as f:
        tasks = json.load(f)
    
    # Init TaskSampler
    task_sampler = TaskSampler(tasks)

    # Load query template
    with open(cfg.query_template_file, 'r') as f:
        template = f.read()

    # Init LLMQueryHandler
    template_name = os.path.basename(cfg.query_template_file).split('.')[0]
    save_dir = os.path.join(cfg.save_root, cfg.llm, f"{cfg.num_constraints}types", template_name)
    os.makedirs(save_dir, exist_ok=True)
    llm_configs = cfg['api_configs'][cfg.llm]
    llm_handler = ConcurrentLLMQueryHandler(
        model_name=llm_configs['name'],
        api_key=llm_configs['key'],
        model_provider=llm_configs['provider'],
        output_file_prefix='resps',
        save_dir=save_dir,
        k_samples=cfg.k_samples,
        temperature=cfg.temperature,
        top_p=cfg.top_p
    )
    llm_handler.num_types = cfg.num_constraints
    
    # Processing
    print(f"Processing Model {llm_configs['provider']}-{cfg.llm} for types = {cfg.num_constraints}") 
    llm_handler.process_tasks(task_sampler, template,
                              reset_progress=cfg.get('reset_progress', False),
                              first_k=cfg.get('first_k', None))
    
    # --- Evaluate samples ---
    with open(llm_handler.output_file, 'r') as f:
        responses = json.load(f)['results']

    acc_metrics = compute_acc_metric(verify_by_thresholded_difficulty, responses, threshold=4)
    acc_metrics = sorted(acc_metrics, key=lambda x: x[-1], reverse=True)
    df = pd.DataFrame(data=acc_metrics, columns=['task_id', 'constraint_id', 'pass@k', 'acc_var'])
    df.to_csv(os.path.join(save_dir, 'k_sample_metrics.csv'), index=False)
