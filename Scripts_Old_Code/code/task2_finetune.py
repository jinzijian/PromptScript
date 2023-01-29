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

# args
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--epochs", type=int, default=0, help="learning rate")
parser.add_argument("--batch_size", type=int, default=500, help="batch_size")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--eps", type=float, default=1e-8, help="adam_epsilon")
parser.add_argument("--seed", type=int, default=0, help="adam_epsilon")
parser.add_argument("--cpu", type=bool, default=False, help="use cpu")
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
train_datas, train_labels = get_all_tasktwo_dataset(train_path)
test_datas, test_labels = get_all_tasktwo_dataset(test_path)
dev_datas, dev_labels = get_all_tasktwo_dataset(dev_path)
train_dataset = owndataset(train_datas, train_labels)
test_dataset = owndataset(test_datas, test_labels)
dev_dataset = owndataset(dev_datas, dev_labels)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

# model & tokenizer & optimizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.parameters(), lr=args.lr)
num_training_steps = args.epochs * len(train_loader)
lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
)
model.train()
for epoch in range(args.epochs):
    model.train()
    # todo： text和labels 都需要padding么？这个转化为tensor的标准流程是咋样的
    for events, labels in train_loader:
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, events)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        labels_ids = labels
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        outputs = model(input_ids=input_ids, labels=labels_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print('/n')
        print(outputs.loss)

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
        labels_ids = labels
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        outputs = model(input_ids=input_ids, labels=labels_ids)
        pred_logits = outputs.logits
        bz = len(input_ids)
        for i in range(bz):
            pred_ids = torch.argsort(pred_logits[i], dim=0, descending=True)
            if pred_ids[0] == labels[i].item():
                hit += 1
            if pred_ids[0] == labels[i].item() and pred_ids[0] == 1:
                tp += 1
            if pred_ids[0] == labels[i].item() and pred_ids[0] == 0:
                tn += 1
            if pred_ids[0] != labels[i].item() and pred_ids[0] == 1:
                fp += 1
            if pred_ids[0] != labels[i].item() and pred_ids[0] == 0:
                fn += 1
            all += 1
print(" dev accuracy is " + str(hit / all))
print(" dev evaluate " + str(all) + " tp " + str(tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn))

print('finish dev')

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
        labels_ids = labels
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        outputs = model(input_ids=input_ids, labels=labels_ids)
        pred_logits = outputs.logits
        bz = len(input_ids)
        for i in range(bz):
            pred_ids = torch.argsort(pred_logits[i], dim=0, descending=True)
            if pred_ids[0] == labels[i].item():
                hit += 1
            if pred_ids[0] == labels[i].item() and pred_ids[0] == 1:
                tp += 1
            if pred_ids[0] == labels[i].item() and pred_ids[0] == 0:
                tn += 1
            if pred_ids[0] != labels[i].item() and pred_ids[0] == 1:
                fp += 1
            if pred_ids[0] != labels[i].item() and pred_ids[0] == 0:
                fn += 1
            all += 1

print(" test accuracy is " + str(hit / all))
print(" test evaluate " + str(all) + " tp " + str(tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn))

print('finish test')
torch.save(model, str(args.lr) + " " + str(args.epochs) + " task2_finetune_checkpoint.pkl")