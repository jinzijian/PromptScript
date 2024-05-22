import os, sys
from openai import OpenAI
import pandas as pd
import pickle
import core

def make_client(cfg):
    llm_cfg = cfg[cfg.run.llm_type]
    client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=llm_cfg.api_key
    )
    if not cfg.run.llm_type in ['gpt3', 'gpt4']:
        client.base_url = llm_cfg.base_url
    
    return client
    

def make_content(cfg, q_string):
    prompt = 'Translate the following from Chinese to English: <T>. Do not add any extra output.' 
    
    content = prompt.replace('<T>', q_string)
    return content


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
    )
    return chat_completion.choices[0].message.content

def translate(cfg, df, client):
    fields = ['task', 'skill_constraints', 'item_constraints', 
              'environment_constraints', 'time_constraints']
    tl_df = pd.DataFrame(columns=fields)
    
    def _clean_dict(s):
        if isinstance(s, dict):
            return list(s.values())[0]
        return s
    
    for i, row in df.iterrows():
        # if i < 305:
        #     continue
        tl_row = {}
        for f in fields:
            if f == 'task':
                tl_row['task'] = get_response(cfg, client, 
                                              make_content(cfg, _clean_dict(row[f]))
                                              )
            else:
                items = eval(row[f])
                tl_items = []
                for itm in items:
                    tl_items.append(get_response(cfg, client, 
                                                 make_content(cfg, _clean_dict(itm))
                                                 )
                                    )
                tl_row[f] = [tl_items]
        tl_row = pd.DataFrame.from_dict(tl_row)
        tl_df = pd.concat([tl_df, tl_row], ignore_index=True)
        tl_df.to_json('./topic_constraints_en_tab_2403.json', orient='index', indent=4)
        print(f'Row {i} is translated')
    
    return tl_df
        
def save_list_to_file(my_list, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(my_list, file)


if __name__ == '__main__':
    args = core.config.get_args()
    cfg = core.config.get_config(args.cfg_file)
    
    core.tools.makedir_exist_ok(os.path.join(cfg.run.save_path, cfg.run.llm_type))
    
    client = make_client(cfg)
    
    df_zh = pd.read_csv('./data/topic_constraints_zh_preprocessed_2403.csv')
    df_en = translate(cfg, df_zh, client)
    df_en.to_csv('./data/topic_constraints_en_2403.csv')