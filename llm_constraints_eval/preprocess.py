import pandas as pd
import json
import re
import ast

def clean_json_string(s):
    s = s.replace('“', '"').replace('”', '"')
    s = s.replace('_x000D_','')
    s = re.sub(r'\[\“', '[', s)
    s = re.sub(r',\s*\]', ']', s)  # Remove consecutive commas, basic scenario
   
    return s

def parse_row(constraints):
    def _parse_time_constraints(c):
        tc = c.get('时间限制')
        if not isinstance(tc, str):
            try:
                tc = tc[0]
            except:
                tc = tc.get('描述')
                assert tc is not None
        # print(tc)
        return tc
    
    def _parse_constraints(c):
        if not isinstance(c[0], str):
            c = [c[i].get(f'限制{i+1}') for i in range(len(c))]
        return [c]
        
    # print(len(constraints.get('限制').get('技能限制')))
    if constraints.get('限制', None) is not None:
        new_row = {
                'task': constraints.get('主题', ''),
                'skill_constraints': _parse_constraints(constraints.get('限制').get('技能限制')),
                'item_constraints': _parse_constraints(constraints.get('限制').get('物品限制')),
                'environment_constraints': _parse_constraints(constraints.get('限制').get('场景限制')),
                'time_constraints': [[_parse_time_constraints(constraints.get('限制'))]]
            }
    else:
        new_row = {
                'task': constraints.get('主题', ''),
                'skill_constraints': _parse_constraints(constraints.get('技能限制')),
                'item_constraints': _parse_constraints(constraints.get('物品限制')),
                'environment_constraints': _parse_constraints(constraints.get('场景限制')),
                'time_constraints': [[_parse_time_constraints(constraints)]]
            }
    return new_row

def parse_env_constraints(csv_path):
    df = pd.read_csv(csv_path)

    if 'env_constraints' not in df.columns:
        raise ValueError("The CSV does not have an 'env_constraints' column.")
   
    constraints_df = pd.DataFrame(columns=['task', 'skill_constraints', 'item_constraints', 'environment_constraints', 'time_constraints', 
                                          's1','s2','i1','i2','e1','e2'])
   
    r = 0
    for _, df_row in df.iterrows():

        row = df_row['env_constraints']
        row = clean_json_string(row)
        # print(row)
        constraints = json.loads(row)
        new_row = parse_row(constraints)

        _df_row = pd.DataFrame.from_dict(new_row)

        constraints_df = pd.concat([constraints_df, _df_row], ignore_index=True)
        r += 1
        if r == 500:
            break
    
    constraints_df[['s1','s2','i1','i2','e1','e2']] = df[:500][['s1','s2','i1','i2','e1','e2']].copy()
    return constraints_df


if __name__ == '__main__':
    parsed_df = parse_env_constraints('./data/topic_constraints_zh_2403.csv')
    parsed_df.to_csv('./data/topic_constraints_zh_parsed_2403.csv')