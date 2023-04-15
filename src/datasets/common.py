import numpy as np
import joblib
import random
import itertools
from sentence_transformers import InputExample
from tqdm import tqdm
from sklearn.model_selection import KFold

def load_preprocessed(path):
    data = joblib.load(path)
    return data['train'], data['test']


def load_formatted_data(path):
    return joblib.load(path)


def get_positive_pairs(k2v, mode='index', key_texts=None, val_texts=None, flatten=False):
    if flatten:
        pairs = []
    else:
        pairs = {}
    for k in k2v.keys():
        if mode == 'index':
            p = itertools.product([k], k2v[k], [1])
        elif mode == 'text':
            k_txt, v_txt = [key_texts[k]], [val_texts[v] for v in k2v[k]]
            p = itertools.product(k_txt, v_txt, [1])
        elif mode == 'sbert':
            index_pairs = itertools.product([k], k2v[k])
            p = [InputExample(texts=(key_texts[p[0]], val_texts[p[1]]), label=1) for p in index_pairs]
        if flatten:
            pairs.extend(p)
        else:
            pairs[k] = list(p)
    return pairs

def get_negative_pairs(k2v, total_v=None, neg_size=-1, seed=0, mode='index', key_texts=None, val_texts=None, flatten=False):
    if total_v is None:
        all_v = []
        for v in k2v.values(): all_v.extend(v)
        all_v = set(all_v)
    else:
        all_v = set(np.arange(0, total_v, dtype=int))
    if flatten:
        pairs = []
    else:
        pairs = {}
    for k in tqdm(k2v.keys(), desc='Processing pairs', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
        other_i = list(all_v - set(k2v[k]))
        if neg_size != -1 and len(other_i) > len(k2v[k]) * neg_size:
            np.random.seed(seed + k)
            other_i = np.random.choice(other_i, size=len(k2v[k]) * neg_size, replace=False)
        if mode == 'index':
            p = itertools.product([k], other_i, [0])
        elif mode == 'text':
            t_txt, i_txt = [key_texts[k]], [val_texts[i] for i in other_i]
            p = itertools.product(t_txt, i_txt, [0])
        elif mode == 'sbert':
            index_pairs = itertools.product([k], other_i)
            p = [InputExample(texts=(key_texts[p[0]], val_texts[p[1]]), label=0) for p in index_pairs]
        if flatten:
            pairs.extend(p)
        else:
            pairs[k] = list(p)
    return pairs

def get_pairs(k2v, total_v=None, neg_size=-1, mode='index', key_texts=None, val_texts=None, seed=0):
    pos_pairs = get_positive_pairs(k2v, mode, key_texts, val_texts, flatten=True)
    neg_pairs = get_negative_pairs(k2v, total_v, neg_size, seed, mode, key_texts, val_texts, flatten=True)
    neg_pairs.extend(pos_pairs)
    return neg_pairs

def sample_pairs(pos_pairs, neg_pairs, sample_topics=None, neg_size=-1, seed=0):
    pairs = []
    if sample_topics is None:
        sample_topics = pos_pairs.keys()
    for t in sample_topics:
        sampled_pos = pos_pairs[t]
        sampled_neg = neg_pairs[t]
        pairs.extend(sampled_pos)
        if neg_size != -1 and len(sampled_neg) > len(sampled_pos) * neg_size:
            np.random.seed(seed)
            perm = np.random.permutation(len(sampled_neg))
            for i in perm[:int(len(sampled_pos) * neg_size)]:
                pairs.append(sampled_neg[i])
        else:
            pairs.extend(sampled_neg)
    return pairs

def sub_dict(d, keys):
    subd = {}
    for k in keys: 
        subd[k] = d[k]
    return subd

def k_fold_split(k2v, n_splits=5, seed=0):
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    train_data, test_data = {}, {}
    keys = list(k2v.keys())
    for i, (train_k, test_k) in enumerate(kf.split(keys)):
        # train_data[i] = get_pairs(sub_dict(k2v, train_k), neg_size, seed)
        # test_data[i] = get_pairs(sub_dict(k2v, test_k))
        train_data[i] = sub_dict(k2v, train_k)
        test_data[i] = sub_dict(k2v, test_k)
    return train_data, test_data

def k_fold_split_pairs(pos_pairs, neg_pairs, n_splits=5, neg_size=-1, seed=0):
    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    train_data, test_data = {}, {}
    all_k = np.arange(len(pos_pairs), dtype=int)
    for i, (train_t, test_t) in enumerate(kf.split(all_k)):
        train_data[i] = sample_pairs(pos_pairs, neg_pairs, train_t, neg_size, seed + i)
        test_data[i] = sample_pairs(pos_pairs, neg_pairs, test_t)
    return train_data, test_data
        
# def make_index_pairs(n, k, m):
#     m = int(m)
#     pairs = [(i, i) for i in range(n)]
#     pairs_diff = []
#     for i in range(n):
#         if i == 0:
#             sample = np.random.randint(1, n, size=min(n-1, m))
#         elif i == n - 1:
#             sample = np.random.randint(0, n-1, size=min(n-1, m))
#         else:
#             left = np.random.randint(0, i, size=min(i, m))
#             right = np.random.randint(i+1, n, size=min(n-i-1, m))
#             sample = np.random.permutation(np.append(left, right))[:m]
#         pairs_diff.extend([(i, j) for i, j in zip([i] * m, sample)])
#     pairs.extend(pairs_diff)
#     return pairs

# def get_positive_pairs(data, sbert_type=False):
#     pairs = []
#     for topic, items in tqdm(data.items(), desc='Processing positive pairs', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
#         for item in items:
#             if sbert_type:
#                 pairs.append(InputExample(texts=[topic, item], label=1))
#             else:
#                 pairs.append((topic, item))
#     return pairs

# def get_negative_pairs(data, k=-1, seed=0, sbert_type=False):
#     np.random.seed(seed)
#     items = flatten_items(data)
#     all_item_idx = set(np.arange(0, len(items)))
#     pairs = []
#     cur_topic_idx = 0
#     for topic in tqdm(data.keys(), desc='Processing negative pairs', bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
#         topic_item_idx = set(np.arange(cur_topic_idx, cur_topic_idx + len(data[topic]), dtype=int))
#         cur_topic_idx = cur_topic_idx + len(data[topic])
#         idx = list(all_item_idx - topic_item_idx)
#         if k != -1 and len(idx) > len(data[topic]) * k:
#             idx = np.random.choice(idx, size=len(data[topic]) * k, replace=False)
#         for i in idx:
#             if sbert_type:
#                 pairs.append(InputExample(texts=[topic, items[i]], label=0))
#             else:
#                 pairs.append((topic, items[i]))
#     return pairs

def sbert_data_to_lists(data):
    a = [example.texts[0] for example in data]
    b = [example.texts[1] for example in data]
    labels = [int(example.label) for example in data]
    return a, b, labels

def split_data(data, test_ratio, seed=0):
    np.random.seed(seed)
    n_test = int(len(data) * test_ratio)
    data = np.random.permutation(data)
    return data[n_test:], data[:n_test]

def sbert_data_class_ratio(data):
    labels = [int(example.label) for example in data]
    return np.mean(labels)
