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
from modeling import PTuneForLAMA
# args
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=1, help="gpu")
parser.add_argument("--epochs", type=int, default=20, help="learning rate")
parser.add_argument("--batch_size", type=int, default=500, help="batch_size")
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
test_path = "/home/zijian/Scripts_new/zxy_test"
dev_path = "/home/zijian/Scripts_new/all_data_temp_dev"
all_data_path = "/home/zijian/Scripts_new/all_data_temp"
train_datas, train_labels = get_all_taskthree_dataset(train_path)
test_datas, test_labels = get_all_taskthree_dataset(test_path)
dev_datas, dev_labels = get_all_taskthree_dataset(dev_path)
train_dataset = owndataset(train_datas, train_labels)
test_dataset = owndataset(test_datas, test_labels)
dev_dataset = owndataset(dev_datas, dev_labels)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)

# model & tokenizer & optimizer
model = PTuneForLAMA(device, args.template).to(device)
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
        labels_ids, labels_token = convert_text_to_ids(tokenizer, labels)
        labels_ids = seq_padding(tokenizer, labels_ids)
        input_ids, token_type_ids, labels_ids = input_ids.long(), token_type_ids.long(), labels_ids.long()
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels_ids = labels_ids.to(device)
        loss,logits,hit = model(input_ids, labels_ids,len(input_ids))
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        print('/n')
        print(loss)
torch.save(model, str(args.lr) + " " + str(args.epochs) + " task3_ptuning_checkpoint.pkl")

text1 = 'before'
tokenized_text1 = tokenizer.tokenize(text1)
indexed_tokens_before = tokenizer.convert_tokens_to_ids(tokenized_text1)
# exclude number
text2 = 'after'
tokenized_text2 = tokenizer.tokenize(text2)
indexed_tokens_after = tokenizer.convert_tokens_to_ids(tokenized_text2)
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
        loss,logits,hit = model(input_ids, labels_ids,len(input_ids))
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
            if pred_logit[masked_index][indexed_tokens_before] > pred_logit[masked_index][indexed_tokens_after]:
                pred_id = 'before'
            else:
                pred_id = 'after'
            result = tokenized_label_text[masked_index]
            if pred_id == result:
                hit += 1
            if pred_id == result and pred_id == 'before':
                tp += 1
            if pred_id == result and pred_id == 'after':
                tn += 1
            if pred_id != result and pred_id == 'before':
                fp += 1
            if pred_id != result and pred_id == 'after':
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
#model = torch.load("/home/zijian/Scripts_new/5e-05 20 task3_ptuning_checkpoint.pkl").to(device)
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
        loss,logits,hit = model(input_ids, labels_ids,len(input_ids))
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
            if pred_logit[masked_index][indexed_tokens_before] > pred_logit[masked_index][indexed_tokens_after]:
                pred_id = 'before'
            else:
                pred_id = 'after'
            result = tokenized_label_text[masked_index]
            if pred_id == result:
                hit += 1
            if pred_id == result and pred_id == 'before':
                tp += 1
            if pred_id == result and pred_id == 'after':
                tn += 1
            if pred_id != result and pred_id == 'before':
                fp += 1
            if pred_id != result and pred_id == 'after':
                fn += 1
            all += 1
print(" test accuracy is " + str(hit / all))
print(" test evaluate " + str(all) + " tp " + str(tp) + ' tn ' + str(tn) + " fp " + str(fp) + " fn " + str(fn))

print('finish test')
