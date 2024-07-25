import numpy
import sys, os
import joblib
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import core

PATH = './outputs/gpt3/evals'
RESP_FILE = './outputs/gpt3/all_resps_gpt3.json'
SAVE_FILE = './outputs/gpt3/all_evals_gpt3.json'
SAVE_SCORE_FIG = './outputs/gpt3/scores.jpg'

# PATH = './outputs/vicuna/evals'
# RESP_FILE = './outputs/vicuna/all_resps_vicuna.json'
# SAVE_FILE = './outputs/vicuna/all_evals_vicuna.json'
# SAVE_SCORE_FIG = './outputs/vicuna/scores.jpg'

def connect_all_files(path):
    with open(RESP_FILE, 'r') as f:
        all_resps = json.load(f)
    files = glob.glob(os.path.join(path, '*.pkl'))
    files = sorted(files, key=lambda f: int(f.split('_')[-1].split('.')[0]))

    all_evals = {}
    for f in files:
        rows = joblib.load(f)
        for r in rows:
            all_evals[str(r[0])] = {}
            for k, v in all_resps[str(r[0])].items():
                all_evals[str(r[0])][k] = v
            all_evals[str(r[0])]['gpt4_eval'] = r[1]
    return all_evals

def parse_scores(all_resps):
    all_scores = {'0':[], '1':[], '2':[], '3':[], '4':[], 'total': []}
    for resp_id, resp in all_resps.items():
        scores = eval(resp['gpt4_eval'])
        tot = 0
        # for i in scores.keys():
        #     all_scores[str(i)].append(int(scores[i]))
        #     tot += int(scores[i])
        for i in range(len(scores)):
            all_scores[str(i)].append(int(scores[i]))
            tot += int(scores[i])
        all_scores['total'].append(tot)
            
    for i in all_scores:
        all_scores[i] = np.array(all_scores[i])
    return all_scores
        
def bin_counts(x, bins=None, bin_range=None, nbins=100):
    if x is None or len(x) == 0:
        return None, None
    
    if bins is None:
        if bin_range is None:
            bin_range = (np.min(x), np.max(x) + (np.max(x) - np.min(x)) / nbins)
        bin_edges = np.linspace(bin_range[0], bin_range[1], nbins+1)
    else:
        bin_edges = bins
        nbins = len(bins) - 1
    counts = np.zeros(nbins)
    bin_indices = np.digitize(x, bin_edges) - 1
    
    for j in range(nbins):
        counts[j] = np.count_nonzero(bin_indices == j)

    return bin_edges, counts

def flatten_axes(axes):
    axes_flat = []
    if not hasattr(axes, 'ndim'):
        axes_flat.append(axes)
    elif axes.ndim == 1:
        for ax in axes:
            axes_flat.append(ax)
    else:
        for axs in axes:
            for ax in axs:
                axes_flat.append(ax)
    return axes_flat

# all_resps = connect_all_files(PATH)
# with open(SAVE_FILE, 'w', encoding='utf8') as f:
#     json.dump(all_resps, f, ensure_ascii=False, indent=4, separators=(',', ': '))

# all_scores = parse_scores(all_resps)

# mean_score = np.mean(all_scores['total'])

# ncols = 2
# nrows = np.ceil(len(all_scores) / ncols).astype(int)
# fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
# axes_flat = flatten_axes(axes)
# bins = np.array([4, 8, 12, 16, 20, 24])
# score_names = ['Relevance', 'Accuracy', 'Thoroughness', 'Clarity', 'Rationality / Cost']
# for i, s in enumerate(all_scores.keys()):
#     if s == 'total': continue
#     ax = axes_flat[i]
#     _, cnts = bin_counts(all_scores[s], bins=bins)
#     # ax.step(bins[:-1], cnts, where='post', color='b')
#     # ax.fill_between(bins[:-1], cnts, step='post', label=f'Score_{s}', alpha=0.7, color='b')
#     ax.bar(bins[:-1], cnts, label=f'Score_{score_names[i]}')
#     ax.set_xticks(bins[:-1])
#     ax.legend()
#     ax.set_xlabel('Score')
#     ax.set_ylabel('Counts')

# bins = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
# _, cnts = bin_counts(all_scores['total'], bins=bins)
# axes_flat[-1].bar(bins[:-1], cnts, label=f'Total (mean={mean_score:.2f})')
# axes_flat[-1].legend()
# fig.tight_layout()
# fig.savefig(SAVE_SCORE_FIG)

def make_constraints(content, types, seed):
    msg = ''
    rates = []
    for t in range(len(types)):
        key = f'{types[t]}_constraints'
        np.random.seed(seed)
        i = np.random.randint(0, len(content[key]))
        if types[t] == 'skill':
            # print(content[key][i], content[f's{i+1}'])
            rate = content[f's{i+1}']
            rates.append((f's{i+1}', rate))
        elif types[t] == 'item':
            rate = content[f'i{i+1}']
            rates.append((f'i{i+1}', rate))
        elif types[t] == 'environment':
            rate = content[f'e{i+1}']
            rates.append((f'e{i+1}', rate))
        
        if len(msg) > 0:
            msg = msg + ' '
        msg = msg + f'({t+1}) ' + content[key][i]
    return msg, rates

def giveup_rates(rates, resps):
    rate_avg, rate_std = [], []
    for lvl in [1, 2, 3, 4, 5]:
        lvl_idx = np.array(rates) == lvl
        resp_lvl = np.array(resps)[lvl_idx]
        if len(resp_lvl) > 1:
            avg, std = np.mean(resp_lvl), np.std(resp_lvl)
        elif len(resp_lvl) == 1: 
            avg, std = resp_lvl[0], 0.
        else:
            avg, std = 0., 0.
        rate_avg.append(avg)
        rate_std.append(std)
    return rate_avg, rate_std

def plot_mean_std(fpath, mu1, std1, mu2=None, std2=None):
    def _plot(mu, std, color):
        plt.plot(mu, '-*', color=color)
        upper_bound = [m + s for m, s in zip(mu, std)]
        lower_bound = [m - s for m, s in zip(mu, std)]
        plt.fill_between(np.arange(len(mu)), y1=lower_bound, y2=upper_bound, color=color, alpha=0.25)
        plt.xlabel('Constraint Levels')
        plt.ylabel('Giveup Rate')
        plt.xticks([0, 1, 2, 3, 4,])
        plt.ylim((0, 1))
        # plt.ylim((-10, 10))
    
    _plot(mu1, std1, 'r')
    if mu2 is not None:
        _plot(mu2, std2, 'b')
    plt.savefig(fpath)

def plot_bars(fpath, levels, vals_dict):
    width = 0.25 / len(vals_dict)
    levels = np.array(levels)
    i = 0
    for k, vals in vals_dict.items():
        offsets = (i - len(vals_dict) / 2) * width + width / 2
        plt.bar(levels + offsets, vals, width, label=k)
        plt.ylim((0, 1))
        i += 1
    plt.legend()
    constraints = fpath.split('_')[1].split('=')[-1]
    plt.title(f'Give-up Rate under Constraints={constraints}')
    plt.xlabel('Constraints Level')
    plt.ylabel('Give-up Rate')
    plt.savefig(fpath)

def get_giveup_rates(data_path, eval_path, types, seed):
    with open(os.path.join(data_path), 'r') as f:
        data = json.load(f)
        
    with open(os.path.join(eval_path), 'r') as f:
        evals = json.load(f)
    
    START = 0
    END = len(data.keys())
    levels = []
    results = []
    ref_results = []
    
    def _parse_judgement(x):
        if '0,' in x:
            return 0
        elif '1,' in x:
            return 1
        else:
            return None
        
    for i in range(START, END):
        # print(i)
        constraints, rates = make_constraints(data[f'{i}'], types, seed)
        levels.append(max([r[1] for r in rates]))
        p0 = _parse_judgement(evals[f'{i}']['resps'])
        # p1 = _parse_judgement(evals[f'{i}']['ref_resps'])
        # if p0 is not None and p1 is not None:
        if p0 is not None:
            results.append(p0)
            # ref_results.append(p1)
        # print(evals[f'{i}']['ref_resps'][1])
        
        # if i % 10 == 0 or i == END - 1:
        #     save_responses(cfg, resp_buffer)
    # fpath = './results/figs/gpt3_type=environment_seed=0_giveup_lvl.jpg'
    valids = []
    for i in range(len(results)):
        try:
            int(results[i])
            valids.append(i)
        except:
            pass
    
    levels = np.array(levels)[valids]
    results = np.array(results)[valids]
    # ref_results = np.array(ref_results)[valids]
    results = [int(r) for r in results]
    # ref_results = [int(r) for r in ref_results]
    avgs, stds = giveup_rates(levels, results)
    # ref_avgs, ref_stds = giveup_rates(levels, ref_results)
    
    # return avgs, stds, ref_avgs, ref_stds
    return avgs, stds

def group_by_level(data_path, eval_path, output_path, types, seed):
    with open(os.path.join(data_path), 'r') as f:
        data = json.load(f)
        
    with open(os.path.join(eval_path), 'r') as f:
        evals = json.load(f)
    
    START = 0
    END = len(data.keys())
        
    for i in range(START, END):
        # print(i)
        constraints, rates = make_constraints(data[f'{i}'], types, seed)
        evals[f'{i}']['constraint_id'] = [r[0] for r in rates]
        evals[f'{i}']['constraint_level'] = [r[1] for r in rates]
    
    with open(os.path.join(output_path), 'w') as f:
        json.dump(evals, f, indent=4)    

if __name__ == '__main__':
    # args = core.config.get_args()
    # cfg = core.config.get_config(args.cfg_file)

    types = ['skill','item','environment']
    # types = ['environment']
    seed = 0
    TYPE = '-'.join(types) if len(types) > 1 else types[0]
    dpath = './data/topic_constraints_en_2403_v2.json'
    # vicuna_path = './results/eval=vicuna-gpt4_types=environment_seed=0.json'
    # gpt3_path = './results/eval=gpt3-gpt4_types=environment_seed=0.json'
    vicuna_path = f'./results/vicuna_types={TYPE}_seed=0.json'
    gpt3_path = f'./results/gpt3_types={TYPE}_seed=0.json'
    gpt4_path = f'./results/gpt4_types={TYPE}_seed=0.json'
    gpt4_moe_path = f'./results/gpt4_types={TYPE}_seed=0moe.json'
    
    fpath = f'./results/figs/eval_0724/eval_types={TYPE}_seed=0_giveup_lvl.jpg'
    gpt3_avgs, gpt3_stds = get_giveup_rates(dpath, gpt3_path, types, seed)
    gpt4_avgs, gpt4_stds = get_giveup_rates(dpath, gpt4_path, types, seed)
    vicuna_avgs, vicuna_stds = get_giveup_rates(dpath, vicuna_path, types, seed)
    gpt4moe_avgs, gpt4moe_stds = get_giveup_rates(dpath, gpt4_moe_path, types, seed)
    plot_bars(fpath, [1, 2, 3, 4, 5], 
              {'gpt4': gpt4_avgs, 'gpt3': gpt3_avgs, 'vicuna': vicuna_avgs, 'gpt4-moe': gpt4moe_avgs})
    
    # group_by_level(dpath, gpt4_moe_path, f'./results/gpt4_types={TYPE}_seed=0moe_new.json', types, seed)