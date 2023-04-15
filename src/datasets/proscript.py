import numpy as np
import joblib
import torch
import itertools
from .common import *
from torch.utils.data import Dataset

def get_items_and_topics(dataset):
    items = [] # all items
    t2i = {} # topic to item index mapping
    for sample in dataset:
        topic = sample["topic"]
        if topic not in t2i:
            t2i[topic] = []
        for sequence in sample["text"]:
            for item in sequence:
                items.append(item)
                t2i[topic].append(len(items) - 1)
    topics = list(t2i.keys()) # all topics
    i2t = {} # item to topic index mapping
    for i, t in enumerate(topics):
        t2i[i] = t2i.pop(t)
        for j in t2i[i]:
            if j not in i2t:
                i2t[j] = []
            i2t[j].append(i)
    return topics, items, t2i, i2t


def get_items_by_topic(dataset):
    items = {}
    for sample in dataset:
        topic = sample["topic"]
        for sequence in sample["text"]:
            for item in sequence:
                if topic not in items:
                    items[topic] = []
                items[topic].append(item)
    return items


def preprocess_data(path, 
                    splits_path=None,
                    split_level='topic', 
                    n_folds=5, 
                    seed=0):
    data = load_formatted_data(path)
    topics, items, t2i, i2t = get_items_and_topics(data)
    if splits_path is not None:
        train_folds, test_folds = load_preprocessed(splits_path)
    else:
        if split_level == 'both' or split_level == 'topic':
            train_folds, test_folds = k_fold_split(t2i, n_splits=n_folds, seed=seed)
        elif split_level == 'item':
            train_folds, test_folds = k_fold_split(i2t, n_splits=n_folds, seed=seed)
    return topics, items, t2i, i2t, train_folds, test_folds


class ProscriptDataset(Dataset):
    def __init__(self, topics, items, pairs, item_first=False, mode='sbert'):
        self.topics = topics
        self.items = items
        self.pairs = pairs
        self.item_first = item_first
        self.mode = mode
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        if self.item_first:
            i, t, label = self.pairs[idx]
        else:
            t, i, label = self.pairs[idx]
        if self.mode == 'sbert':
            return InputExample(texts=(self.topics[t], self.items[i]), label=label)
        elif self.mode == 'index':
            return (t, i, label)
        elif self.mode == 'text':
            return [self.topics[t], self.items[i], label]


def get_dataset(data, topics, items, mode='sbert', split_level='both', neg_size=-1, seed=0):
    total_v = None
    if split_level == 'topic': total_v = len(items)
    pairs = get_pairs(data, total_v=total_v, neg_size=neg_size, mode='index', seed=seed)
    return ProscriptDataset(topics, items, pairs, item_first=split_level=='item', mode=mode)
