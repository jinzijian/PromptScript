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

text1 = 'include'
tokenized_text1 = tokenizer.tokenize(text1)
indexed_tokens_include = tokenizer.convert_tokens_to_ids(tokenized_text1)
# exclude number
text2 = 'except'
tokenized_text2 = tokenizer.tokenize(text2)
indexed_tokens_except = tokenizer.convert_tokens_to_ids(tokenized_text2)
main_event = ["make hot dog"]
candidates = ["take the hot dog from the freezer .", "defrost the hot dog ." ,"put the hot dog in the oven .",
              "wait for some time .", "take out the hot dog .", "serve the hot dog with bread .",
              "add ketchup on the top ."]
datas = []
for c in candidates:
    datas.append(main_event[0] + " [MASK] " + c)
res = []
for data in datas:
    inputs = tokenizer(data, return_tensors="pt").to(device)
    with torch.no_grad():
        masked_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        pred_logit = include_model(**inputs).logits[0][masked_index]
        l1 = pred_logit[0][indexed_tokens_include]
        l2 = pred_logit[0][indexed_tokens_except]
        if pred_logit[0][indexed_tokens_include] > pred_logit[0][indexed_tokens_except]:
            pred_id = 'include'
        else:
            pred_id = 'except'
        res.append(pred_id)
print(res)