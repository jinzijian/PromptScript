from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import RobertaPreTrainedModel, XLMRobertaModel
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from typing import Optional, Tuple
import torch.nn as nn
import torch.nn.functional as func
class MarginLoss(nn.Module):
    def __init__(self, device,margin = 1):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.device = device
        return

    def forward(self, logits, labels):
        loss = torch.tensor([0.0], requires_grad=True).to(self.device)
        res = -1
        for i in range(len(labels)):
            if labels[i].item() == 1:
                res = i
        max_score = logits[res]
        for l in logits:
            tmp = max(l - max_score + self.margin, torch.tensor([0.000001], requires_grad=True).to(self.device))
            loss = loss + tmp
        loss = loss / len(logits)
        return loss

def get_embedding_layer(model):
    embeddings = model.get_input_embeddings()
    return embeddings
def get_vocab_by_strategy(args, tokenizer):
    return tokenizer.get_vocab()

def token_wrapper(args, token):
    return token


class start_pmodel(torch.nn.Module):
    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # load pre-trained model
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embeddings = get_embedding_layer(self.model)

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()
        self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device, args)
        self.prompt_encoder = self.prompt_encoder.to(self.device)
        self.classifier = nn.Linear(768, 1)


    # TODO get query 完全不懂
    def get_query(self, input_ids):
        text1 = 'start'
        tokenized_text1 = self.tokenizer.tokenize(text1)
        start_id  = self.tokenizer.convert_tokens_to_ids(tokenized_text1)
        pid = self.tokenizer.unk_token_id
        bz = input_ids.size(dim=0)
        res = []
        for i in range(bz):
            input_id = input_ids[i]
            length = input_id.size(dim = 0)
            tmp_id = input_id
            tmp_id = tmp_id.cpu().numpy().tolist()
            index = tmp_id.index(start_id[0])
            end_index = tmp_id.index(102)
            tmp_id[index] = pid
            tmp_id[index + 1] = pid
            tmp_id = tmp_id[:index + 2] + [pid] + tmp_id[index + 2:]
            new_id = tmp_id[0:1] + [pid] * 3 + tmp_id[1:end_index+1] + [pid] * 3 + tmp_id[end_index+1:]
            res.append((new_id))
        res = np.array(res)
        res = torch.from_numpy(res).to(self.device)
        return res





    def embed_input(self, input_ids):
        input_ids = self.get_query(input_ids)
        bz = input_ids.size(dim = 0)
        queries_for_embedding = input_ids.clone()
        raw_embeds = self.embeddings(queries_for_embedding)
        # For using handcraft prompts
        ls = (input_ids == self.tokenizer.unk_token_id).nonzero()
        #print(ls.size())
        blocked_indices = (input_ids == self.tokenizer.unk_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        replace_embeds = self.prompt_encoder()
        for bidx in range(bz):
            for i in range(self.prompt_encoder.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds


    def forward(self,
                input_ids = None,
                labels = None,
                attention_mask = None,
                return_dict=True,
                ):
        bz = len(input_ids)
        # construct query ids
        # get embedded input
        inputs_embeds = self.embed_input(input_ids)
        outputs = self.model(inputs_embeds=inputs_embeds,
                             return_dict = True)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, :1, :]
        sequence_output = sequence_output.squeeze()
        logits = self.classifier(sequence_output)
        if bz == 1:
            logits = torch.unsqueeze(logits, 0)
        loss_fct = MarginLoss(self.device)
        loss = loss_fct(logits, labels)
        return loss, logits


class start_finetune_pmodel(torch.nn.Module):
    def __init__(self, args, device, template):
        super().__init__()
        self.args = args
        self.device = device

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # load pre-trained model
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # set allowed vocab set
        self.classifier = nn.Linear(768, 1)




    def forward(self,
                input_ids = None,
                labels = None,
                attention_mask = None,
                return_dict=True,
                ):
        bz = len(input_ids)
        # construct query ids
        # get embedded input
        outputs = self.model(input_ids =  input_ids,
                             return_dict = True)
        sequence_output = outputs[0]
        sequence_output = sequence_output[:, :1, :]
        sequence_output = sequence_output.squeeze()
        logits = self.classifier(sequence_output)
        if bz == 1:
            logits = torch.unsqueeze(logits, 0)
        loss_fct = MarginLoss(self.device)
        loss = loss_fct(logits, labels)
        return loss, logits



class PromptEncoder(torch.nn.Module):
    def __init__(self, template, hidden_size, tokenizer, device, args):
        super().__init__()
        self.device = device
        self.spell_length = sum(template)
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template
        self.cloze_mask = [
            [1] * self.cloze_length[0]  # first cloze
            + [1] * self.cloze_length[1]  # second cloze
            + [1] * self.cloze_length[2]  # third cloze
        ]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(self.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(self.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(self.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=0.3,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self):
        input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds