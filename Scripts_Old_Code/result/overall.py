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
parser.add_argument("--mode", type=str, default="ptuning", help="different settings")
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
    start_model = torch.load("/home/zijian/Scripts_new/5e-05 0 margin_task2_finetune_checkpoint.pkl").to(device)
    order_model = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
if args.mode == 'finetune':
    include_model = torch.load("/home/zijian/Scripts_new/5e-05 1010negative_size100.0 task1_finetune_checkpoint.pkl").to(device)
    start_model = torch.load("/home/zijian/Scripts_new/5e-05 20 margin_task2_finetune_checkpoint.pkl").to(device)
    order_model = torch.load("/home/zijian/Scripts_new/5e-05 20 task3_finetune_checkpoint.pkl").to(device)
if args.mode == 'ptuning':
    include_model = torch.load("/home/zijian/Scripts_new/5e-05 10negative100.0 test_task1_ptuning_checkpoint.pkl").to(device)
    start_model = torch.load("/home/zijian/Scripts_new/5e-05 30 margin_task2_ptuning_checkpoint.pkl").to(device)
    order_model = torch.load("/home/zijian/Scripts_new/5e-05 20 task3_ptuning_checkpoint.pkl").to(device)
if args.mode == 'freeze':
    include_model = torch.load("/home/zijian/Scripts_new/5e-05 10negative100.0 test_task1_ptuning_freeze_checkpoint.pkl").to(device)
    start_model = torch.load("/home/zijian/Scripts_new/5e-05 20 freeze_margin_task2_ptuning_checkpoint.pkl").to(device)
    order_model = torch.load("/home/zijian/Scripts_new/5e-05 20 task3_ptuning_freeze_checkpoint.pkl").to(device)



# dataset
class owndataset():
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label

test_path = "/home/zijian/Scripts_new/all_data_test"
all_data_path = "/home/zijian/Scripts_new/all_data2"

main_events = get_all_main_events(test_path)
print("create all true main_events, number " + str(len(main_events)))
true_sub_events = get_all_true_sub_events(test_path)
print("create all true sub_events, number " + str(len(true_sub_events)))
all_possible_sub_events = get_all_possible_sub_events(all_data_path)
print("create all true sub_events, number " + str(len(all_possible_sub_events)))

# include
res = []
include_data, include_labels = construct_overall_include_data(main_events, all_possible_sub_events)
include_dataset = owndataset(include_data, include_labels)
test_loader = DataLoader(include_dataset, batch_size=args.batch_size, shuffle=False)
text1 = 'include'
tokenized_text1 = tokenizer.tokenize(text1)
indexed_tokens_include = tokenizer.convert_tokens_to_ids(tokenized_text1)
# exclude number
text2 = 'except'
tokenized_text2 = tokenizer.tokenize(text2)
indexed_tokens_except = tokenizer.convert_tokens_to_ids(tokenized_text2)
for events, labels in test_loader:
    with torch.no_grad():
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, events)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        labels_ids, labels_token = convert_text_to_ids(tokenizer, labels)
        labels_ids = seq_padding(tokenizer, labels_ids)
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        if (args.mode != "ptuning") and (args.mode != "freeze") :
            outputs = include_model(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels_ids)
            pred_logits = outputs.logits
        if args.mode == "ptuning" or args.mode == "freeze":
            loss, logits, hit = include_model(input_ids, labels_ids, len(input_ids))
            pred_logits = logits
        bz = len(input_ids)
        for i in range(bz):
            ids = input_ids[i]
            new_id = ids.cpu().numpy().tolist()
            event_text = tokenizer.decode(input_ids[i])
            tokenized_text = tokenizer.tokenize(event_text)
            masked_index = new_id.index(103)
            label_text = tokenizer.decode(labels_ids[i])
            tokenized_label_text = tokenizer.tokenize(label_text)
            pred_logit = pred_logits[i]
            if pred_logit[masked_index][indexed_tokens_include] > pred_logit[masked_index][indexed_tokens_except]:
                pred_id = 'include'
            else:
                pred_id = 'except'
            if pred_id == "include":
                res.append(1)
            else:
                res.append(0)

print('finish evaluating')
print(str(len(res)))
res = np.array(res)
new_res = np.split(res, len(main_events))
include_results = []
for r in new_res:
    tmp = []
    for i in range(len(r)):
        if r[i] == 1:
            tmp.append(all_possible_sub_events[i])
    include_results.append(tmp)
#print(include_results)
def compare2(subs, presubs):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    presubs = list(set(presubs))
    for presub in presubs:
        if(presub in subs):
            tp += 1
    recall = tp/len(subs)
    if(len(presubs) != 0):
        precision = tp/len(presubs)
    else:
        precision = 0
    return recall, precision

# get stage 1 matric
print(include_results)
include_resmatric = []
for i in range(len(include_results)):
    tmp = []
    include_result = include_results[i]
    sub_event = true_sub_events[i]
    recall, precision = compare2(sub_event, include_result)
    tmp.append(recall)
    tmp.append(precision)
    include_resmatric.append(tmp)

len_res = len(include_resmatric)
r = 0
p = 0
for i in range(len(include_resmatric)):
    r += include_resmatric[i][0]
    p += include_resmatric[i][1]
print("average recall is" + str(r / len_res))
print("average precision is" + str(p / len_res))
print(include_resmatric)
# random subs is all true included subevents
random_subs = []
for sub in true_sub_events:
    tmp = copy.deepcopy(sub)
    random.shuffle(tmp)
    random_subs.append(tmp)
#print(random_subs)

labels = []
def consturct_start_data(main_events, include_results):
    startwith_data = []
    for i in range(len(main_events)):
        tmp_labels = []
        start_data = []
        main_event = main_events[i]
        sub_result = include_results[i]
        for j in range(len(sub_result)):
            text = main_event + ' start with ' + sub_result[j]
            tmp_labels.append(0)
            start_data.append(text)
        labels.append(tmp_labels)
        startwith_data.append(start_data)
    return startwith_data

startwith_data = consturct_start_data(main_events, include_results)
startwith_data_only = consturct_start_data(main_events, random_subs)
#print(startwith_data)

def get_start_result(startwith_data, include_results):
    all_logits = []
    for i in range(len(startwith_data)):
        logits = []
        sentences = startwith_data[i]
        for j in range(len(sentences)):
            label = torch.tensor([1]).unsqueeze(0)
            sentence = sentences[j]
            label = label.to(device)
            texts = sentence.split(" ")
            idx = texts.index("start")
            main = " ".join(texts[:idx])
            sub = " ".join(texts[idx + 2:])
            inputs = tokenizer(sentence, return_tensors='pt').to(device)
            with torch.no_grad():
                input_ids = inputs["input_ids"]
                label = torch.tensor(1).unsqueeze(0)
                label_ids = label.to(device)
                loss, outputs = start_model(input_ids, labels=label_ids)
                logit = outputs
            logits.append(logit[0][0].item())
        all_logits.append(logits)
    #print(all_logits)
    all_logits = np.array(all_logits, dtype='object')
    start_results = []
    other_subs = copy.deepcopy(include_results)
    for i in range(len(all_logits)):
        include_result = include_results[i]
        other_sub = other_subs[i]
        if len(include_result) != 0:
            logits = all_logits[i]
            idx = np.argmax(logits)
            start_result = include_result[idx]
            del other_sub[idx]
        else:
            start_result = 'None'
        start_results.append(start_result)
    return start_results, other_subs

start_results, other_subs = get_start_result(startwith_data, include_results)
start_results_only, other_subs_only = get_start_result(startwith_data_only, random_subs)
print(start_results)
true_start = []
for subs in true_sub_events:
    true_start.append(subs[0])
acc = 0
for i in range(len(start_results_only)):
    if(start_results_only[i] == true_start[i]):
        acc += 1
print("start acc")
start_only_acc = acc / len(start_results_only)
print(start_only_acc)

order_firsts = []


# construct data for order
# order_firsts  && other_subs as input
for start_result in start_results:
    order = []
    order.append(start_result)
    order_firsts.append(order)


# right start and right include
true_order_firsts = []
true_other_subs = []
for subs in true_sub_events:
    true_order = []
    true_other_sub = []
    true_order.append(subs[0])
    for i in range(1, len(subs)):
        true_other_sub.append(subs[i])
    true_order_firsts.append(true_order)
    true_other_subs.append(true_other_sub)

# Order all events
text3 = 'before'
tokenized_text3 = tokenizer.tokenize(text3)
indexed_tokens_before = tokenizer.convert_tokens_to_ids(tokenized_text3)


def get_logit(sentence, tokenizer, indexed_tokens):
    tokenized_text = tokenizer.tokenize(sentence)
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
        if(args.mode != 'ptuning' and args.mode != "freeze"):
            outputs_order = order_model(**inputs, labels=order_label)
        else:
            input_ids = inputs["input_ids"]
            loss,logits,hit = order_model(input_ids, order_label, len(input_ids))
    if (args.mode != 'ptuning' and args.mode != "freeze"):
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
final_result = getOrderResult(order_firsts, other_subs)
true_final_results = getOrderResult(true_order_firsts, true_other_subs)


def rouge(a, b):
    rouge = Rouge()
    rouge_score = rouge.get_scores(a, b, avg=True)  # a和b里面包含多个句子的时候用
    #rouge_score1 = rouge.get_scores(a, b)  # a和b里面只包含一个句子的时候用
    # 以上两句可根据自己的需求来进行选择
    r1 = rouge_score["rouge-1"]
    r2 = rouge_score["rouge-2"]
    rl = rouge_score["rouge-l"]

    return r1, r2, rl

def get_avg(results):
    r = 0
    p = 0
    f = 0
    l = len(results)
    for result in results:
        r += result.get('r')
        p += result.get('p')
        f += result.get('f')
    return r/l, p/l, f/l

def get_rougel_result(res, sub_events):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    ress = []
    for r in res:
        str = ""
        for j in r:
            str += " " + j
        ress.append(str)
    print(ress)
    subs = []
    for r in sub_events:
        str = ""
        for j in r:
            str += " " + j
        subs.append(str)
    print(subs)
    r1s = []
    r2s = []
    rls = []
    for i in range(len(ress)):
        a = ress[i]
        b = subs[i]
        scores = scorer.score(a, b)
        dict = {}
        dict['r'] = scores["rougeL"][1]
        dict['p'] = scores["rougeL"][0]
        dict['f'] = scores["rougeL"][2]
        rls.append(dict)
    return rls
res = final_result
rls = get_rougel_result(res, true_sub_events)
true_rls = get_rougel_result(true_final_results, true_sub_events)


avg_rl, avg_pl, avg_fl = get_avg(rls)
avgorder_rl, avgorder_pl, avgorder_fl = get_avg(true_rls)
print(avg_rl, avg_pl, avg_fl)
print(avgorder_rl, avgorder_pl, avgorder_fl)
path_file_name ='/home/zijian/Scripts_new/over_result.txt'
if not os.path.exists(path_file_name):
    fileObject = open(path_file_name, 'a+', encoding='utf-8')
    fileObject.write(
        args.mode + str(avg_rl) + " " +  str(avg_pl) + " " + str(avg_fl) + "order_result" + repr(avgorder_rl) + " " +  repr(avgorder_pl) + " " + repr(avgorder_fl)
    )
    fileObject.write('\n')
    fileObject.close()
else:
    fileObject = open(path_file_name, 'a+', encoding='utf-8')
    fileObject.write(
        args.mode + repr(avg_rl) + " " +  repr(avg_pl) + " " + repr(avg_fl) + "order_result" + repr(avgorder_rl) + " " +  repr(avgorder_pl) + " " + repr(avgorder_fl)
    )
    fileObject.write('\n')
    fileObject.close()




































