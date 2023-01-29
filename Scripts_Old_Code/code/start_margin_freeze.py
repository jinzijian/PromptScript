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
from start_ptuning_freeze import *

# args
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1, help="gpu")
parser.add_argument("--epochs", type=int, default=20, help="learning rate")
parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
parser.add_argument("--eps", type=float, default=1e-8, help="adam_epsilon")
parser.add_argument("--seed", type=int, default=0, help="adam_epsilon")
parser.add_argument("--cpu", type=bool, default=False, help="use cpu")
parser.add_argument("--template", type=str, default="(3, 3, 3)")
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
train_datas, train_labels = get_all_taskfour_dataset(train_path)
test_datas, test_labels = get_all_taskfour_dataset(test_path)
dev_datas, dev_labels = get_all_taskfour_dataset(dev_path)
train_dataset = owndataset(train_datas, train_labels)
test_dataset = owndataset(test_datas, test_labels)
dev_dataset = owndataset(dev_datas, dev_labels)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

# model & tokenizer & optimizer
num_labels = 2
template = [3,3,3]
model = start_pmodel(args, device, template).to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
optimizer = AdamW(model.prompt_encoder.parameters(), lr=args.lr)
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
        events = events
        labels = labels
        e = []
        l = []
        for event in events:
            for ex in event:
                e.append(ex)
        for label in labels:
            for la in label:
                l.append(la)
        events = e
        labels = l
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, events)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        labels_ids = torch.stack(labels)
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        loss, logits = model(input_ids=input_ids, labels=labels_ids)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print('/n')
        print(loss)

hit = 0
all = 0
model.eval()
for events, labels in dev_loader:
    with torch.no_grad():
        events = events
        labels = labels
        e = []
        l = []
        for event in events:
            for ex in event:
                e.append(ex)
        for label in labels:
            for la in label:
                l.append(la)
        events = e
        labels = l
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(events)
        random.seed(randnum)
        random.shuffle(labels)
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, events)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        labels_ids = torch.stack(labels)
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        loss, pred_logits = model(input_ids=input_ids, labels=labels_ids)
        bz = len(input_ids)
        pred_ids = pred_logits.cpu()
        pred_res = np.argmax(pred_ids)
        if labels[pred_res] == 1:
            hit += 1
        all += 1
print(" dev accuracy is " + str(hit / all))
#print(" dev evaluate " + str(all) + " tp " + str(tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn))

print('finish dev')

hit = 0
all = 0
model.eval()
for events, labels in test_loader:
    with torch.no_grad():
        events = events
        labels = labels
        e = []
        l = []
        for event in events:
            for ex in event:
                e.append(ex)
        for label in labels:
            for la in label:
                l.append(la)
        events = e
        labels = l
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(events)
        random.seed(randnum)
        random.shuffle(labels)
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, events)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        labels_ids = torch.stack(labels)
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        loss, pred_logits = model(input_ids=input_ids, labels=labels_ids)
        bz = len(input_ids)
        pred_ids = pred_logits.cpu()
        pred_res = np.argmax(pred_ids)
        if labels[pred_res] == 1:
            hit += 1
        all += 1

print(" test accuracy is " + str(hit / all))
#print(" test evaluate " + str(all) + " tp " + str(tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn))

print('finish test')
torch.save(model, str(args.lr) + " " + str(args.epochs) + " new freeze_margin_task2_ptuning_checkpoint.pkl")
