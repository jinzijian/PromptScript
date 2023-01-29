import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import  xml.dom.minidom
from random import choice
import random
from transformers import BertTokenizer

# helper function
def get_main_event(file_path):
    x = file_path.split("/")
    x = x[-1].split('.')
    flag = 0
    if(len(x) == 3):
        flag = 1
    main_event = x[0]
    main_event = main_event.replace("_", " ")
    return main_event, flag

def get_all_file(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.xml':
                L.append(os.path.join(root, file))
    return L

def shuffle_together(a, b):
    a = np.array(a)
    b = np.array(b)
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    return a, b

def convert_text_to_ids(tokenizer, text, max_len=128):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    elif isinstance(text, tuple):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    else:
        print("Unexpected input")
    return input_ids, token_type_ids

def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    L = [len(x) for x in X]
    ML = max(L)
    X = torch.Tensor([x + [pad_id] * (ML - len(x)) if len(x) < ML else x for x in X])
    return X

# create task1 dataset
def construct_positive_include(file_path):
    main_event, flag = get_main_event(file_path)
    include_text = []
    include_label = []
    dom = xml.dom.minidom.parse(file_path)
    root = dom.documentElement
    gold_labels = root.getElementsByTagName('script') # 每个scripts
    for gold_label in gold_labels:
        sentences = []
        sentence = gold_label.getElementsByTagName("item") # 一个scripts里所有的subevents
        for node in sentence: # node 每个 subevent
            if(flag == 1):
                text = node.getAttribute("original")
            else:
                text = node.getAttribute("text")
            sentences.append(text)
        prompt = " include "
        for sen in sentences:
            sentence = main_event + " [MASK] " + sen
            label = main_event + prompt + sen
            # print(sentence)
            # print(label)
            include_text.append(sentence)
            include_label.append(label)
    return include_text, include_label

def construct_negative_include(all_file_path, file_path):
    files = get_all_file(all_file_path)
    except_text = []
    except_label = []
    main_event, flag = get_main_event(file_path)
    for f in files:
        if f == file_path:
            continue
        dom = xml.dom.minidom.parse(f)
        root = dom.documentElement
        gold_labels = root.getElementsByTagName('script')
        for gold_label in gold_labels:
            sentences = []
            sentence = gold_label.getElementsByTagName("item")
            for node in sentence:
                if (flag == 1):
                    text = node.getAttribute("original")
                else:
                    text = node.getAttribute("text")
                sentences.append(text)
            prompt = " except "
            for sen in sentences:
                sentence = main_event + " [MASK] " + sen
                label = main_event + prompt + sen
                # print(sentence)
                # print(label)
                except_text.append(sentence)
                except_label.append(label)
    return except_text, except_label



def get_all_taskone_dataset(train_file_path, all_data_path, negative_size = 1.0):
    texts = []
    labels = []
    train_files = get_all_file(train_file_path)
    for file in train_files:
        include_text, include_label = construct_positive_include(file)
        except_text, except_label = construct_negative_include(all_data_path, file)
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(except_text)
        random.seed(randnum)
        random.shuffle(except_label)
        except_text = except_text[:int(negative_size * len(include_text))]
        except_label = except_label[:int(negative_size * len(include_text))]
        texts.extend(include_text)
        texts.extend(except_text)
        labels.extend(include_label)
        labels.extend(except_label)
    return texts, labels


# create task2 dataset
def construct_start_onefile(file_path):
    main_event, flag = get_main_event(file_path)
    texts = []
    labels = []
    dom = xml.dom.minidom.parse(file_path)
    root = dom.documentElement
    gold_labels = root.getElementsByTagName('script')
    for gold_label in gold_labels:
        sentences = []
        sentence = gold_label.getElementsByTagName("item")
        # wirte word
        for node in sentence:
            if (flag == 1):
                text = node.getAttribute("original")
            else:
                text = node.getAttribute("text")
            sentences.append(text)
        for i in range(len(sentences)):
            texts.append(main_event + " start with " + sentences[i])
            if(i == 0):
                labels.append(1)
            else:
                labels.append(0)
    return texts, labels

def get_all_tasktwo_dataset(train_file_path):
    texts = []
    labels = []
    train_files = get_all_file(train_file_path)
    for f in train_files:
        start_text, start_label = construct_start_onefile(f)
        texts.extend(start_text)
        labels.extend(start_label)
    return texts, labels


# create task3 dataset
def construct_order_onefile(file_path):
    main_event, flag = get_main_event(file_path)
    events = []
    labels = []
    dom = xml.dom.minidom.parse(file_path)
    root = dom.documentElement
    gold_labels = root.getElementsByTagName('script')
    for gold_label in gold_labels:
        sentences = []
        sentence = gold_label.getElementsByTagName("item")
        for node in sentence:
            if flag == 1:
                text = node.getAttribute("original")
            else:
                text = node.getAttribute("text")
            sentences.append(text)
        prompt = " before "
        # bofore 为 0
        for i in range(len(sentences) - 1):
            j = i + 1
            s = sentences[i][:-1] + ' [MASK] ' + sentences[j]
            label = sentences[i][:-1] + prompt + sentences[j]
            events.append(s)
            labels.append(label)
        prompt = " after "
        # after 为 1
        for i in range(len(sentences) - 1):
            j = i + 1
            s = sentences[j][:-1] + ' [MASK] ' + sentences[i]
            label = sentences[j][:-1] + prompt + sentences[i]
            events.append(s)
            labels.append(label)
    return events, labels



def get_all_taskthree_dataset(train_file_path):
    texts = []
    labels = []
    train_files = get_all_file(train_file_path)
    for f in train_files:
        order_text, order_label = construct_order_onefile(f)
        texts.extend(order_text)
        labels.extend(order_label)
    return texts, labels

# task 4
def construct_start_onescript(file_path):
    main_event, flag = get_main_event(file_path)
    texts_res = []
    labels_res = []
    dom = xml.dom.minidom.parse(file_path)
    root = dom.documentElement
    gold_labels = root.getElementsByTagName('script')
    for gold_label in gold_labels:
        sentences = []
        sentence = gold_label.getElementsByTagName("item")
        # wirte word
        for node in sentence:
            if (flag == 1):
                text = node.getAttribute("original")
            else:
                text = node.getAttribute("text")
            sentences.append(text)
        texts = []
        labels = []
        for i in range(len(sentences)):
            texts.append(main_event + " start with " + sentences[i])
            if (i == 0):
                labels.append(1)
            else:
                labels.append(0)
        texts_res.append(texts)
        labels_res.append(labels)
    return texts_res, labels_res

def get_all_taskfour_dataset(train_file_path):
    texts = []
    labels = []
    train_files = get_all_file(train_file_path)
    for f in train_files:
        start_text, start_label = construct_start_onescript(f)
        texts.extend(start_text)
        labels.extend(start_label)
    return texts, labels

def get_all_main_events(file_path):
    main_events = []
    all_files = get_all_file(file_path)
    for file in all_files:
        main_event, _ = get_main_event(file)
        main_events.append(main_event)
    return  main_events

def get_all_true_sub_events(file_path):
    sub_events = []
    all_files = get_all_file(file_path)
    for file in all_files:
        main_event, flag = get_main_event(file)
        dom = xml.dom.minidom.parse(file)
        root = dom.documentElement
        gold_labels = root.getElementsByTagName('script')
        for gold_label in gold_labels:
            sentences = []
            sentence = gold_label.getElementsByTagName("item")
            for node in sentence:
                if flag == 1:
                    text = node.getAttribute("original")
                else:
                    text = node.getAttribute("text")
                sentences.append(text)
            sub_events.append(sentences)
    return sub_events

def get_all_possible_sub_events(file_path):
    sub_events = []
    all_files = get_all_file(file_path)
    sentences = []
    for file in all_files:
        main_event, flag = get_main_event(file)
        dom = xml.dom.minidom.parse(file)
        root = dom.documentElement
        gold_labels = root.getElementsByTagName('script')
        for gold_label in gold_labels:
            sentence = gold_label.getElementsByTagName("item")
            for node in sentence:
                if flag == 1:
                    text = node.getAttribute("original")
                else:
                    text = node.getAttribute("text")
                sentences.append(text)
            sub_events.append(sentences)
    return sentences

def construct_overall_include_data(main_events, all_possible_sub_events):
    include_data = []
    include_labels = []
    for main_event in main_events:
        for all_possible_sub_event in all_possible_sub_events:
            s = main_event + " [MASK] " + all_possible_sub_event
            include_data.append(s)
            include_labels.append(s)
    return include_data, include_labels




if __name__ == '__main__':
    ds = "../datas/pilot_esd/baking a cake.pilot.xml"
    ds_all = "../datas/pilot_esd"
    om = "../datas/OMICS/answer telephone.xml"
    main_event, flag = get_main_event(ds)
    include_text, include_label = construct_positive_include("../datas/OMICS/answer telephone.xml")
    except_text, except_label = construct_negative_include("../datas/OMICS", "../datas/OMICS/answer telephone.xml")
    # taskone_texts, taskone_labels = get_all_taskone_dataset("../datas/OMICS", "../datas/OMICS")
    # tasktwo_texts, tasktwo_labels = get_all_tasktwo_dataset("../datas/OMICS")
    # orders, labels = construct_order_onefile("../datas/OMICS/answer telephone.xml")
    # taskthree_texts, taskthree_labels = get_all_taskthree_dataset("../datas/OMICS")
    taskone_texts, taskone_labels = get_all_taskone_dataset(ds_all, ds_all)
    tasktwo_texts, tasktwo_labels = get_all_tasktwo_dataset(ds_all)
    orders, labels = construct_order_onefile(ds)
    taskthree_texts, taskthree_labels = get_all_taskthree_dataset(ds_all)
    print('pass')

