from rouge import Rouge
import copy
import os
import xml.dom.minidom
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification,BertModel
from dataset import *
import logging
logging.basicConfig(level=logging.INFO)
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, random_split
import random
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification, AdamW, get_scheduler
import numpy as np
import argparse
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from rouge import Rouge
from rouge_score import rouge_scorer

# args
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1, help="gpu")
parser.add_argument("--epochs", type=int, default=0, help="learning rate")
parser.add_argument("--batch_size", type=int, default=500, help="batch_size")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--eps", type=float, default=1e-8, help="adam_epsilon")
parser.add_argument("--seed", type=int, default=0, help="adam_epsilon")
parser.add_argument("--cpu", type=bool, default=False, help="use cpu")
parser.add_argument("--mode", type=str, default="zero", help="different settings")
args = parser.parse_args()

# set seed
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

# set device
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)
    device = args.gpu
if args.cpu:
    device = torch.device('cpu')

# set model & tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
if args.mode == 'zero':
    include_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
    start_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    order_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
if args.mode == 'finetune':
    include_model = torch.load("/home/zijian/Scripts_new/5e-05 10 task1_finetune_checkpoint.pkl").to(device)
    start_model = torch.load("/home/zijian/Scripts_new/5e-05 20 task2_finetune_checkpoint.pkl").to(device)
    order_model = torch.load("/home/zijian/Scripts_new/5e-05 20 task3_finetune_checkpoint.pkl").to(device)
if args.mode == 'ptuning':
    include_model = torch.load("/home/zijian/Scripts_new/5e-05 1 test_task1_ptuning_checkpoint.pkl").to(device)
    start_model = torch.load("/home/zijian/Scripts_new/5e-05 20 task2_ptuning_checkpoint.pkl").to(device)
    order_model = torch.load("/home/zijian/Scripts_new/5e-05 20 task3_ptuning_checkpoint.pkl").to(device)


order_first = [["find your audience ."]]
other_subs = [[ "find a storybook .", "read the storybook .","read out the words .","watch the response from the audience .", "put away the storybook .", "ask the audience if they want more ."]]
text3 = 'before'
tokenized_text3 = tokenizer.tokenize(text3)
indexed_tokens_before = tokenizer.convert_tokens_to_ids(tokenized_text3)


def get_logit(sentence, tokenizer, indexed_tokens):
    tokenized_text = tokenizer.tokenize(sentence)
    print(tokenized_text)
    order_label = tokenizer(sentence, return_tensors="pt")['input_ids']
    order_label = order_label.to(device)
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = inputs.to(device)
    input_ids = inputs["input_ids"].cpu()
    ids = input_ids.numpy()
    index = np.argwhere(ids == 103)
    masked_index = index[0][1]
    with torch.no_grad():
        #print("orderding")
        if(args.mode != 'ptuning'):
            outputs_order = order_model(**inputs, labels=order_label)
        else:
            input_ids = inputs["input_ids"]
            loss,logits,hit = order_model(input_ids, order_label, len(input_ids))
    if (args.mode != 'ptuning'):
        pred_logits = outputs_order.logits
    else:
        pred_logits = logits
    logit = pred_logits[0][masked_index][indexed_tokens].item()
    return logit

def update_lists(order_first, other_sub, res):
    res = np.array(res)
    idx = np.argmax(res)
    next_event = other_sub[idx]
    order_first.append(next_event)
    del other_sub[idx]
    return

def find_next(order_firsts, other_subs):
    for i in range(len(order_firsts)):
        order_first = order_firsts[i]
        res = []
        start_result = order_first[-1]
        other_sub = other_subs[i]
        if len(other_sub) == 0:
            continue
        else:
            for j in range(len(other_sub)):
                sub = other_sub[j]
                sentence = start_result + ' [MASK] ' + sub
                #print("get logit ing")
                logit = get_logit(sentence=sentence, tokenizer=tokenizer, indexed_tokens=indexed_tokens_before)
                res.append(logit)
            #print("update")
            update_lists(order_first, other_sub, res)
    return

def getOrderResult(order_firsts, other_subs):
    times = []
    length = len(other_subs)
    for i in range(len(other_subs)):
        # print("current")
        # print(i)
        # print("all ")
        # print(length)
        other_sub = other_subs[i]
        times.append(len(other_sub))
    #print(times)

    max_time = np.max(np.array(times))
    #print(max_time)
    for i in range(max_time):
        find_next(order_firsts, other_subs)
    return order_firsts
final_result = getOrderResult(order_first, other_subs)
print(final_result)

#
# def rouge(a, b):
#     rouge = Rouge()
#     rouge_score = rouge.get_scores(a, b, avg=True)  # a和b里面包含多个句子的时候用
#     #rouge_score1 = rouge.get_scores(a, b)  # a和b里面只包含一个句子的时候用
#     # 以上两句可根据自己的需求来进行选择
#     r1 = rouge_score["rouge-1"]
#     r2 = rouge_score["rouge-2"]
#     rl = rouge_score["rouge-l"]
#
#     return r1, r2, rl
# def get_avg(results):
#     r = 0
#     p = 0
#     f = 0
#     l = len(results)
#     for result in results:
#         r += result.get('r')
#         p += result.get('p')
#         f += result.get('f')
#     return r/l, p/l, f/l
#
# def get_rougel_result(res, sub_events):
#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
#     ress = []
#     for r in res:
#         str = ""
#         for j in r:
#             str += " " + j
#         ress.append(str)
#     print(ress)
#     subs = []
#     for r in sub_events:
#         str = ""
#         for j in r:
#             str += " " + j
#         subs.append(str)
#     print(subs)
#     r1s = []
#     r2s = []
#     rls = []
#     for i in range(len(ress)):
#         a = ress[i]
#         b = subs[i]
#         scores = scorer.score(a, b)
#         rls.append(scores["rougeL"][2])
#     return rls
#
# a1 = [['gather dirty clothes .', "turn on dryer .", "put clothes in dryer .","when washing machine is done take out clothes .","turn on washing machine .", "fill washing machine with dirty clothes ."]]
# b1 = [['gather dirty clothes .', 'when washing machine is done take out clothes .', 'fill washing machine with dirty clothes .', 'turn on washing machine .', 'put clothes in dryer .', 'turn on dryer .']]
# rl = get_rougel_result(a1, b1)
# print(get_rougel_result(a1, b1))