import os, sys
import pandas as pd
import pickle
import core
import json
import numpy as np
import core
from task_sampler import *
from query import *
from evaluation import *

def main(cfg):
    for stage in cfg.stages:
        if stage == 'preprocess':
            pass

        elif stage == 'query':
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
            api_configs = cfg['api_configs'][cfg.llm]
            llm_handler = LLMQueryHandler(
                model_name=api_configs['name'],
                api_key=api_configs['key'],
                model_provider=api_configs['provider'],
                output_file_prefix='resps',
                save_dir=save_dir
            )

            llm_handler.num_types = cfg.num_constraints  # Set the num_types
            print(f"Processing Model {api_configs['provider']}-{cfg.llm} for types = {cfg.num_constraints}")
            llm_handler.process_tasks(task_sampler, template, 
                                      reset_progress=cfg.get("reset_progress", False),
                                      first_k=cfg.get("first_k", None))

        elif stage == 'evaluate':
            if cfg.get('result_file', None) is None:
                template_name = os.path.basename(cfg.query_template_file).split('.')[0]
                save_dir = os.path.join(cfg.save_root, cfg.llm, f"{cfg.num_constraints}types", template_name)
                cfg['result_file'] = os.path.join(save_dir, 'resps.json')
            
            df = load_results(cfg['result_file'])
            summary = compute_give_up_rates(df, agg_method=cfg.get("multi_type_agg", 'mean'))

            fig_save_dir = os.path.join(save_dir, 'figs')
            os.makedirs(fig_save_dir, exist_ok=True)
            fig = plot_single_type_give_up_rate(summary['single'])
            plt.savefig(os.path.join(fig_save_dir, 'single_diff_vs_giveup.png'))
            if len(summary['multiple']) > 0:
                plot_multi_type_give_up_rate(summary['multiple'])
                plt.savefig(os.path.join(fig_save_dir, 'multi_diff_vs_giveup.png'))

        elif stage == 'quality_evaluate':
            pass

if __name__ == "__main__":
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    api_configs = core.config.get_config(cfg.api_configs_file)
    cfg['api_configs'] = api_configs

    main(cfg)