from rouge import Rouge
import copy
import os
import xml.dom.minidom
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification, BertModel
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
from modeling import PTuneForLAMA, PTuneForLAMA_freeze

# args
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1, help="gpu")
parser.add_argument("--epochs", type=int, default=10, help="learning rate")
parser.add_argument("--batch_size", type=int, default=500, help="batch_size")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--eps", type=float, default=1e-8, help="adam_epsilon")
parser.add_argument("--seed", type=int, default=0, help="adam_epsilon")
parser.add_argument("--cpu", type=bool, default=False, help="use cpu")
parser.add_argument("--template", type=str, default="(3, 3, 3)")
parser.add_argument("--negative_size", type=float, default=100.0, help="negative_size")
args = parser.parse_args()

print("start code")
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


train_path = "/home/zijian/Scripts_new/all_data_temp_train"
test_path = "/home/zijian/Scripts_new/all_data_temp_test"
dev_path = "/home/zijian/Scripts_new/all_data_temp_dev"
all_data_path = "/home/zijian/Scripts_new/all_data_temp"
train_datas, train_labels = get_all_taskone_dataset(train_path, all_data_path, negative_size=args.negative_size)
test_datas, test_labels = get_all_taskone_dataset(test_path, all_data_path, negative_size=args.negative_size)
dev_datas, dev_labels = get_all_taskone_dataset(dev_path, all_data_path, negative_size=args.negative_size)
train_dataset = owndataset(train_datas, train_labels)
test_dataset = owndataset(test_datas, test_labels)
dev_dataset = owndataset(dev_datas, dev_labels)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

# model & tokenizer & optimizer
model = PTuneForLAMA_freeze(device, args.template).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.prompt_encoder.parameters(), lr=args.lr)
num_training_steps = args.epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
i = 0
model.train()
for epoch in range(args.epochs):
    model.train()
    # todo： text和labels 都需要padding么？这个转化为tensor的标准流程是咋样的
    for events, labels in train_loader:
        i = i + 1
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, events)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        labels_ids, labels_token = convert_text_to_ids(tokenizer, labels)
        labels_ids = seq_padding(tokenizer, labels_ids)
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        loss, logits, hit = model(input_ids, labels_ids, len(input_ids))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        if i == 10:
            break
        print('/n')
        print(loss)

text1 = 'include'
tokenized_text1 = tokenizer.tokenize(text1)
indexed_tokens_include = tokenizer.convert_tokens_to_ids(tokenized_text1)
# exclude number
text2 = 'except'
tokenized_text2 = tokenizer.tokenize(text2)
indexed_tokens_except = tokenizer.convert_tokens_to_ids(tokenized_text2)
hit = 0
tp = 0
tn = 0
fp = 0
fn = 0
all = 0
model.eval()
for events, labels in dev_loader:
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
        loss, logits, hit2 = model(input_ids, labels_ids, len(input_ids))
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
            result = tokenized_label_text[masked_index]
            if pred_id == result:
                hit += 1
            if pred_id == result and pred_id == 'include':
                tp += 1
            if pred_id == result and pred_id == 'except':
                tn += 1
            if pred_id != result and pred_id == 'include':
                fp += 1
            if pred_id != result and pred_id == 'except':
                fn += 1
            all += 1
print(all)
print(hit)

print(" dev accuracy is " + str(hit / all))
print(" dev evaluate " + str(all) + " tp " + str(tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn))

print('finish dev')

path_file_name = '/home/zijian/Scripts_new/result/task1.txt'
if not os.path.exists(path_file_name):
    fileObject = open(path_file_name, 'a+', encoding='utf-8')
    fileObject.write(
        "ptuning" + " dev accuracy is " + str(hit / all) + " dev evaluate " + str(all) + " tp " + str(
            tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn)
    )
    fileObject.write('\n')
    fileObject.close()
else:
    fileObject = open(path_file_name, 'a+', encoding='utf-8')
    fileObject.write(
        "ptuning" + " dev accuracy is " + str(hit / all) + " dev evaluate " + str(all) + " tp " + str(
            tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn)
    )
    fileObject.write('\n')
    fileObject.close()

hit = 0
tp = 0
tn = 0
fp = 0
fn = 0
all = 0
model.eval()
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
        loss, logits, hit2 = model(input_ids, labels_ids, len(input_ids))
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
            result = tokenized_label_text[masked_index]
            if pred_id == result:
                hit += 1
            if pred_id == result and pred_id == 'include':
                tp += 1
            if pred_id == result and pred_id == 'except':
                tn += 1
            if pred_id != result and pred_id == 'include':
                fp += 1
            if pred_id != result and pred_id == 'except':
                fn += 1
            all += 1
print(" test accuracy is " + str(hit / all))
print(" test evaluate " + str(all) + " tp " + str(tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn))

print('finish test')
torch.save(model, str(args.lr) + " " + str(args.epochs) + "negative" + str(
    args.negative_size) + " newtest_task1_ptuning_nfreeze_checkpoint.pkl")

path_file_name = '/home/zijian/Scripts_new/result/task1.txt'
if not os.path.exists(path_file_name):
    fileObject = open(path_file_name, 'a+', encoding='utf-8')
    fileObject.write(
        "test_ptuning" + " test accuracy is " + str(hit / all) + " test evaluate " + str(all) + " tp " + str(
            tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn)
    )
    fileObject.write('\n')
    fileObject.close()
else:
    fileObject = open(path_file_name, 'a+', encoding='utf-8')
    fileObject.write(
        "test_ptuning" + " test accuracy is " + str(hit / all) + " test evaluate " + str(all) + " tp " + str(
            tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn) + "negative_size" + str(args.negative_size)
    )
    fileObject.write('\n')
    fileObject.close()