#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/12/29 1:08 下午
# @Author  : Gear
import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join

import re

import torch.nn as nn

from transformers import AutoTokenizer
#
# from models import get_embedding_layer, create_model
from data_utils.vocab import get_vocab_by_strategy, token_wrapper
from data_utils.dataset import load_file
# from prompt_encoder import PromptEncoder


from transformers import GPT2LMHeadModel, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, BertModel, BertForMaskedLM
#
#
##def create_model(args):
##	if '11b' in args.model_name:
##		from ..megatron_11b.megatron_wrapper import load_megatron_lm
##		print(
##			"Warning: loading MegatronLM (11B) in fp16 requires about 28G GPU memory, and may need 3-5 minutes to load.")
##		return load_megatron_lm(args)
##	MODEL_CLASS, _ = get_model_and_tokenizer_class(args)
##	model = MODEL_CLASS.from_pretrained(args.model_name)
##	if not args.use_lm_finetune:
##		if 'megatron' in args.model_name:
##			raise NotImplementedError("MegatronLM 11B is not for fine-tuning.")
##		model = model.half()
##	return model
##
##
##def get_model_and_tokenizer_class(args):
##	if 'gpt' in args.model_name:
##		return GPT2LMHeadModel, AutoTokenizer
##	elif 'bert' in args.model_name:
##		return BertForMaskedLM, AutoTokenizer
##	elif 'megatron' in args.model_name:
##		return None, AutoTokenizer
##	else:
##		raise NotImplementedError("This model type ``{}'' is not implemented.".format(args.model_name))
#
#
#def get_embedding_layer(args, model):
#	embeddings = model.bert.get_input_embeddings()
#	return embeddings


class PromptEncoder(torch.nn.Module):
	def __init__(self, template, hidden_size, tokenizer, device):
		super().__init__()
		self.device = device
		self.spell_length = 9
		self.hidden_size = hidden_size
		self.tokenizer = tokenizer
		# ent embedding
		template = (3,3,3)
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
		                               dropout=0.0,
		                               bidirectional=True,
		                               batch_first=True)
		self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
		                              nn.ReLU(),
		                              nn.Linear(self.hidden_size, self.hidden_size))
	
	def forward(self):
		input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
		output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
		return output_embeds


class PTuneForLAMA(torch.nn.Module):
	
	def __init__(self, device, template):
		super().__init__()
		self.device = device
		# load tokenizer
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		
		# load pre-trained model
		self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
		self.model = self.model.to(self.device)
#		for param in self.model.parameters():
#		 	param.requires_grad = False
		self.embeddings =  self.model.bert.get_input_embeddings()
		
		# set allowed vocab set
		self.vocab = self.tokenizer.get_vocab()
		#self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
		self.template = template
		
		# load prompt encoder
		self.hidden_size = self.embeddings.embedding_dim
		self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
		self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
		
		self.spell_length = 9
		self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device)
		self.prompt_encoder = self.prompt_encoder.to(self.device)
	
	def embed_input(self, queries, batch):
		bz = batch
		
		queries_for_embedding = queries.clone()
		# print(queries)
		queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
		# print("queries_for_embedding:")
		# print(self.tokenizer.unk_token_id)
		# print(queries_for_embedding)
		raw_embeds = self.embeddings(queries_for_embedding)
		
		# For using handcraft prompts
		
		blocked_indices = (queries == self.pseudo_token_id).nonzero()
		# print(blocked_indices)
		blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
		# print(blocked_indices)
		replace_embeds = self.prompt_encoder()
		# print(replace_embeds.shape)
		for bidx in range(bz):
			for i in range(self.prompt_encoder.spell_length):
				# print(bidx)
				raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
		return raw_embeds
	
	def get_query(self, x_h, prompt_tokens, batch ):
		# For using handcraft prompts
#		if self.args.use_original_template:
#			if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
#				query = re.sub(r'\[Y\].*', '', self.relation_templates[self.args.relation_id].replace('[X]', x_h))
#				return self.tokenizer(' ' + query)['input_ids']
#			else:
#				query = self.relation_templates[self.args.relation_id].replace('[X]', x_h).replace('[Y]',
#				                                                                                   self.tokenizer.mask_token)
#				return self.tokenizer(' ' + query)['input_ids']
#		# For P-tuning
#		if 'gpt' not in self.args.model_name and 'megatron' not in self.args.model_name:
			# BERT-style model
#		tokenized_text = self.tokenizer.tokenize(x_h)
#			#print(tokenized_text)
#			# print(tokenized_text)
#		masked_index = tokenized_text.index(x_t)
#			#print(masked_index)
#			# print(masked_index)
#		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
#			# tokenized_text()
##			query = [
##			         indexed_tokens[0:masked_index] +
##			         prompt_tokens * self.template[0] +
##			         [self.tokenizer.mask_token_id] +
##			         prompt_tokens * self.template[1] +
##			         indexed_tokens[masked_index + 1:]]
#		query = [
#			   indexed_tokens[0:masked_index] +
#			   prompt_tokens * self.template[0] +
#			   [self.tokenizer.mask_token_id] +
#			   prompt_tokens * self.template[1] +
#			   indexed_tokens[masked_index + 1:] +
#			   prompt_tokens * self.template[2]
#			   + [self.tokenizer.sep_token_id]]
		
		
#			print("input_querys:")
#			print(query)
			# tensor = torch.Tensor(query)
		bz = batch
		seq = x_h.cpu().numpy().tolist()
		#print(seq)
		mask_token = 103
		template = (3,3,3)
#		masked_index = seq.index(mask_token)
		queries = []
		seq_token = 102
		for i in range(bz):
			masked_index = seq[i].index(mask_token)
			seq_index =  seq[i].index(seq_token)
			query =[
				seq[i][0:masked_index] + 
				prompt_tokens * template[0] +
				[seq[i][masked_index]] + 
				prompt_tokens * template[1] +
				seq[i][masked_index + 1: seq_index] + 
		     	prompt_tokens * template[2] +
		     	[seq[i][seq_index]] +
		     	seq[i][seq_index+1:]
			]
			#print(query)
			queries.append(query)
			
		return queries
		
		# [[self.tokenizer.cls_token_id]  # [CLS]
		#  + prompt_tokens * self.template[0]
		#  + [self.tokenizer.mask_token_id]  # head entity
		#  + prompt_tokens * self.template[1]
		#  + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
		#  + (prompt_tokens * self.template[2] if self.template[
		#                                             2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
		#  + [self.tokenizer.sep_token_id]
#		
#		elif 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
#			# print('in gpt:')
#			# print(x_h)
#			# GPT-style models
#			return [prompt_tokens * self.template[0]
#			        + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # head entity
#			        + prompt_tokens * self.template[1]
#			        + (self.tokenizer.convert_tokens_to_ids(
#				self.tokenizer.tokenize(' ' + x_t)) if x_t is not None else [])
#			        ]
#		else:
#			raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))
	
#	def get_labels(x_hs, self):
#		# BERT-style model
#		print('bert modelpapapa')
#		# print(x_hs)
#		tokenized_text = self.tokenizer.tokenize(x_hs)
#		# print(tokenized_text)
#		masked_index = tokenized_text.index('before')
#		# print(masked_index)
#		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
#		# tokenized_text()
#		query = [[self.tokenizer.cls_token_id] +
#		         indexed_tokens[0:masked_index] +
#		         [-100 * self.template[0]] +
#		         [self.tokenizer.mask_token_id] +
#		         [-100 * self.template[1]] +
#		         indexed_tokens[masked_index:-1] +
#		         (-100 * self.template[2] if self.template[
#			                                     2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
#		         + [self.tokenizer.sep_token_id]]
#		# print("querys:")
#		# print(query)
#		# tensor = torch.Tensor(query)
#		return query
	def get_labels(self, x_h, labels, batch):
		bz = batch
		seq = x_h.cpu().numpy().tolist()
		labels = labels.cpu().numpy().tolist()
		#print(seq)
		mask_token = 103
		template = (3,3,3)
#		masked_index = seq.index(mask_token)
		queries = []
		seq_token = 102
		for i in range(bz):
			masked_index = seq[i].index(mask_token)
			seq_index =  seq[i].index(seq_token)
			query =[
				labels[i][0:masked_index] + 
				[-100] * template[0] +
				[labels[i][masked_index]] + 
				[-100] * template[1] +
				labels[i][masked_index + 1: seq_index] + 
		     	[-100] * template[2] +
		     	[labels[i][seq_index]] +
		     	labels[i][seq_index+1:]
			]
			#print(query)
			queries.append(query)
			
		return queries
		
		
	def forward(self, events, labels,batch):
		# print(labels)
		bz = batch
		temp_events = events
		# print(events)
		prompt_tokens = [self.pseudo_token_id]
#		seq_labels = labels
#		# x_temp = x_hs
#		queries = torch.Tensor(self.get_query(events, prompt_tokens, x_t=labels))
		
		events = torch.LongTensor(self.get_query(events, prompt_tokens, batch))
		events = events.squeeze(1).to(self.device)
		#print(events.shape)
		#labels = self.get_labels().to(device)
		#print(labels.shape)
		inputs_embeds = self.embed_input(events,batch)
		#print(inputs_embeds.shape)
		labels = torch.LongTensor(self.get_labels(temp_events, labels,  batch)).to(self.device)
		labels = labels.squeeze(1).to(self.device)
		#print(labels.shape)
		#labels = self.get_labels(labels)
		
		outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)
		
		
		
		mask_token = 103
		hit = 0 
		
		for i in range(bz):
			seq = events[i].cpu().numpy().tolist()
			masked_index = seq.index(mask_token)
			
			#print(masked_index)
			prompt = labels[i][masked_index]
			pred_id =  torch.argsort(outputs.logits[i], dim=1, descending=True)
			if pred_id[masked_index][0] == prompt:
				hit += 1
		masked_index = 103
		
		#print(outputs.logits.shape)
		return outputs.loss, outputs.logits,hit
		#print(queries.shape)

#		print("queries:")
#		print(queries)
		
		# print(queries)
		# queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
		# print(queries.shape)
		# print("queries:")
		# print(queries)
		# print("x_t:")
		# print(x_ts)
		# construct label ids
		# label_id =  torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts))
		# print(label_id.shape)
		# label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
		# (bz, -1)).to(self.device)
		# label_ids = torch.Tensor(self.tokenizer.convert_tokens_to_ids(x_ts)).to(self.device)
		# tokenized_text2 = self.tokenizer.tokenize(x_ts)
		# label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_text2)).to(self.device)
		# print('label_id:')
		# print(label_ids)
		# attention_mask = queries != self.pad_token_id
		
		# get embedded input
		queries = queries.to(self.device)
		inputs_embeds = self.embed_input(queries)
		# print(inputs_embeds)
		# print(inputs_embeds1.shape)
		# print(inputs_embeds1)
		
		# print(x_hs)
		# xs = self.get_labels(x_hs)
#		tokenized_text = self.tokenizer.tokenize(events)
		tokens = ['[CLS]'] + tokenizer.tokenize(events) + ['[SEP]']
		
		
		# print(tokenized_text)
		masked_index = tokens.index(labels)
		label_tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

#		query = [
#		         tokens[0:masked_index] +
#		         [-100] * self.template[0] +
#		         [tokens[masked_index]] +
#		         [-100] * self.template[1] +
#		         indexed_tokens[masked_index + 1:-1] +
#		         [-100] * self.template[2] +
#		         [self.tokenizer.sep_token_id]
#		        ]
		

		
		# print("test_model:")
		# print(labels)
		# print(tokenized_text)
		# print(masked_index)
		# print(masked_index)
		# pred_token = self.tokenizer.tokenize('before')
		# print(masked_index)
		#indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		
		# print(indexed_tokens)
		# print(len(indexed_tokens))
		# tokenized_text()
#		query = [
#		         indexed_tokens[0:masked_index] +
#		         [-100] * self.template[0] +
#		         [indexed_tokens[masked_index]] +
#		         [-100] * self.template[1] +
#		         indexed_tokens[masked_index + 1:] +
#		         [-100] * 3 +
#		         [self.tokenizer.sep_token_id]]
		query_labels = torch.Tensor(label_tokens_id).to(self.device)
		#query_labels = pad_sequence(query_labels, True, padding_value=self.pad_token_id).long().to(self.device)
		#print("query_labels:")
		#print(query_labels)
#		print("labels:")
#		print(labels)
		# batch ,
		# INPUT:A [MASK] B
		# LABEL A C B
		outputs = self.model(inputs_embeds=inputs_embeds, labels=query_labels)
		# print(outputs.loss)
		# print(outputs.logits)
		# print(outputs.logits.shape)
		pred_ids = torch.argsort(outputs.logits, dim=2, descending=True)

		text1 = 'before'
		tokenized_text1 = self.tokenizer.tokenize(text1)
		indexed_tokens_before = self.tokenizer.convert_tokens_to_ids(tokenized_text1)
#		print("before token:")
#		print(indexed_tokens_before)

		text2 = 'after'
		tokenized_text2 = self.tokenizer.tokenize(text2)
		indexed_tokens_after = self.tokenizer.convert_tokens_to_ids(tokenized_text2)
#		print("after token:")
#		print(indexed_tokens_after)

		
#		pred_logits = outputs.logits
##
##		print(pred_logits.shape)
#		tokenized_text = self.tokenizer.tokenize(events)
#		masked_index = tokenized_text.index(labels)
#		print("masked_index:")
#		print(masked_index)
#		print(len(tokenized_text))
#		print(indexed_tokens_before)
#		print(indexed_tokens_after)
#		if seq_labels == 'before':
#			print("before")
#			if pred_logits[0][masked_index + 3][indexed_tokens_before] >= pred_logits[0][masked_index + 3][indexed_tokens_after]:
#				hit = 1
#			else:
#				hit = 0
#		
#		elif seq_labels == 'after':
#			print("after")
#			if pred_logits[0][masked_index + 3][indexed_tokens_before] < pred_logits[0][masked_index + 3][indexed_tokens_after]:
#				hit = 1
#			else:
#				hit = 0
		
		#
		# print(pred_ids.shape)
		# # print(pred_ids[0, masked_index, 0])
		# # print(pred_ids[0,masked_index,:][:20])
		# # print()
		# print(indexed_tokens[masked_index])
		# print(pred_ids[0, masked_index + 3, 0])
		print(indexed_tokens_before)
		print(indexed_tokens_after)
		print(indexed_tokens[masked_index])
		
		if indexed_tokens[masked_index] == pred_ids[0, masked_index + 3, 0]:
			hit = 1
		else:
			hit = 0
		return outputs.loss, hit, outputs.logits
# def bert_out():
#     print('inbert')
#     label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(
#         1).to(self.device)  # bz * 1
#
#     labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
#     print(labels)
#     labels = labels.scatter_(1, label_mask, label_ids)
#     output = self.model(**inputs, labels=labels)
#     loss, logits = output.loss, output.logits
#
#     pred_ids = torch.argsort(logits, dim=2, descending=True)
#     hit1 = 0
#     top10 = []
#     for i in range(bz):
#         pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
#         for pred in pred_seq:
#             if pred in self.allowed_vocab_ids:
#                 break
#         if pred == label_ids[i, 0]:
#             hit1 += 1
#
#     if return_candidates:
#         return loss, hit1, top10
#     return loss, hit1
#
# def gpt_out():
#     bz = 1
#     labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
#     label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
#     print("label_mask:")
#     print(label_mask)
#     print(labels)
#     labels = labels.scatter_(1, label_mask, label_ids)
#     #labels[0][2] = torch.tensor(20)
#     #print(inputs_embeds.shape)
#     #inputs_embeds_temp = inputs_embeds[0]
#     #print(inputs_embeds_temp.shape)
#
#     output = self.model(inputs_embeds=inputs_embeds.to(self.device).half(),
#                         attention_mask=attention_mask.to(self.device).half(),
#                         labels=labels.to(self.device))
#     loss, logits = output.loss, output.logits
#
#     #print(loss)
#     #loss.requires_grad = True
#     print(loss)
#     #pred_ids = torch.argsort(logits, dim=2, descending=True)
#     hit1 = 0
#     top10 = []
#     for i in range(bz):
#         top10.append([])
#         pred_seq = pred_ids[i, label_mask[i, 0]].tolist()
#         for pred in pred_seq:
#             if pred in self.allowed_vocab_ids:
#                 top10[-1].append(pred)
#                 if len(top10[-1]) >= 10:
#                     break
#         pred = top10[-1][0]
#         if pred == label_ids[i,0]:
#             hit1 += 1
#
#     if return_candidates:
#         return loss, hit1, top10
#     return loss, hit1
#
# def megatron_out():
#     labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
#     label_mask = (attention_mask.long().sum(dim=1) - 1).unsqueeze(1).to(self.device)
#     labels = labels.scatter_(1, label_mask, label_ids)
#     if not self.args.use_lm_finetune:
#         _attention_mask = attention_mask.float().half()
#         _input_embeds = inputs_embeds.float().half()
#     else:
#         _attention_mask = attention_mask.float()
#         _input_embeds = inputs_embeds.float()
#     output = self.model.decoder.predict(prev_output_tokens=queries,
#                                         inputs_embeds=_input_embeds.to(self.device),
#                                         attention_mask=_attention_mask.to(self.device).bool(),
#                                         labels=labels.to(self.device))
#     logits, loss = output
#
#     pred_ids = torch.argsort(logits, dim=2, descending=True)
#     hit1 = 0
#     top10 = []
#     for i in range(bz):
#         top10.append([])
#         pred_seq = pred_ids[i, label_mask[i, 0]].to(self.device).tolist()
#         for pred in pred_seq:
#             if pred in self.allowed_vocab_ids:
#                 top10[-1].append(pred)
#                 if len(top10[-1]) >= 10:
#                     break
#         pred = top10[-1][0]
#         if pred == label_ids[i, 0]:
#             hit1 += 1
#     if return_candidates:
#         return loss, hit1, top10
#     return loss, hit1

# if 'bert' in self.args.model_name:
#     return bert_out()
# elif 'gpt' in self.args.model_name:
#     return gpt_out()
# elif 'megatron' in self.args.model_name:
#     return megatron_out()
# else:
#     raise NotImplementedError()

class PTuneForLAMA_freeze(torch.nn.Module):

	def __init__(self, device, template):
		super().__init__()
		self.device = device
		# load tokenizer
		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

		# load pre-trained model
		self.model = BertForMaskedLM.from_pretrained("bert-base-uncased")
		self.model = self.model.to(self.device)
		#		for param in self.model.parameters():
		#		 	param.requires_grad = False
		for param in self.model.parameters():
			param.requires_grad = False
		self.embeddings = self.model.bert.get_input_embeddings()

		# set allowed vocab set
		self.vocab = self.tokenizer.get_vocab()
		# self.allowed_vocab_ids = set(self.vocab[k] for k in get_vocab_by_strategy(self.args, self.tokenizer))
		self.template = template

		# load prompt encoder
		self.hidden_size = self.embeddings.embedding_dim
		self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
		self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']

		self.spell_length = 9
		self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.tokenizer, self.device)
		self.prompt_encoder = self.prompt_encoder.to(self.device)

	def embed_input(self, queries, batch):
		bz = batch

		queries_for_embedding = queries.clone()
		# print(queries)
		queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
		# print("queries_for_embedding:")
		# print(self.tokenizer.unk_token_id)
		# print(queries_for_embedding)
		raw_embeds = self.embeddings(queries_for_embedding)

		# For using handcraft prompts

		blocked_indices = (queries == self.pseudo_token_id).nonzero()
		# print(blocked_indices)
		blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
		# print(blocked_indices)
		replace_embeds = self.prompt_encoder()
		# print(replace_embeds.shape)
		for bidx in range(bz):
			for i in range(self.prompt_encoder.spell_length):
				# print(bidx)
				raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
		return raw_embeds

	def get_query(self, x_h, prompt_tokens, batch):
		# For using handcraft prompts
		#		if self.args.use_original_template:
		#			if 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
		#				query = re.sub(r'\[Y\].*', '', self.relation_templates[self.args.relation_id].replace('[X]', x_h))
		#				return self.tokenizer(' ' + query)['input_ids']
		#			else:
		#				query = self.relation_templates[self.args.relation_id].replace('[X]', x_h).replace('[Y]',
		#				                                                                                   self.tokenizer.mask_token)
		#				return self.tokenizer(' ' + query)['input_ids']
		#		# For P-tuning
		#		if 'gpt' not in self.args.model_name and 'megatron' not in self.args.model_name:
		# BERT-style model
		#		tokenized_text = self.tokenizer.tokenize(x_h)
		#			#print(tokenized_text)
		#			# print(tokenized_text)
		#		masked_index = tokenized_text.index(x_t)
		#			#print(masked_index)
		#			# print(masked_index)
		#		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
		#			# tokenized_text()
		##			query = [
		##			         indexed_tokens[0:masked_index] +
		##			         prompt_tokens * self.template[0] +
		##			         [self.tokenizer.mask_token_id] +
		##			         prompt_tokens * self.template[1] +
		##			         indexed_tokens[masked_index + 1:]]
		#		query = [
		#			   indexed_tokens[0:masked_index] +
		#			   prompt_tokens * self.template[0] +
		#			   [self.tokenizer.mask_token_id] +
		#			   prompt_tokens * self.template[1] +
		#			   indexed_tokens[masked_index + 1:] +
		#			   prompt_tokens * self.template[2]
		#			   + [self.tokenizer.sep_token_id]]

		#			print("input_querys:")
		#			print(query)
		# tensor = torch.Tensor(query)
		bz = batch
		seq = x_h.cpu().numpy().tolist()
		# print(seq)
		mask_token = 103
		template = (3, 3, 3)
		#		masked_index = seq.index(mask_token)
		queries = []
		seq_token = 102
		for i in range(bz):
			masked_index = seq[i].index(mask_token)
			seq_index = seq[i].index(seq_token)
			query = [
				seq[i][0:masked_index] +
				prompt_tokens * template[0] +
				[seq[i][masked_index]] +
				prompt_tokens * template[1] +
				seq[i][masked_index + 1: seq_index] +
				prompt_tokens * template[2] +
				[seq[i][seq_index]] +
				seq[i][seq_index + 1:]
			]
			# print(query)
			queries.append(query)

		return queries

	# [[self.tokenizer.cls_token_id]  # [CLS]
	#  + prompt_tokens * self.template[0]
	#  + [self.tokenizer.mask_token_id]  # head entity
	#  + prompt_tokens * self.template[1]
	#  + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # [MASK] (tail entity)
	#  + (prompt_tokens * self.template[2] if self.template[
	#                                             2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
	#  + [self.tokenizer.sep_token_id]
	#
	#		elif 'gpt' in self.args.model_name or 'megatron' in self.args.model_name:
	#			# print('in gpt:')
	#			# print(x_h)
	#			# GPT-style models
	#			return [prompt_tokens * self.template[0]
	#			        + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' ' + x_h))  # head entity
	#			        + prompt_tokens * self.template[1]
	#			        + (self.tokenizer.convert_tokens_to_ids(
	#				self.tokenizer.tokenize(' ' + x_t)) if x_t is not None else [])
	#			        ]
	#		else:
	#			raise NotImplementedError("The query template for {} has not been defined.".format(self.args.model_name))

	#	def get_labels(x_hs, self):
	#		# BERT-style model
	#		print('bert modelpapapa')
	#		# print(x_hs)
	#		tokenized_text = self.tokenizer.tokenize(x_hs)
	#		# print(tokenized_text)
	#		masked_index = tokenized_text.index('before')
	#		# print(masked_index)
	#		indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
	#		# tokenized_text()
	#		query = [[self.tokenizer.cls_token_id] +
	#		         indexed_tokens[0:masked_index] +
	#		         [-100 * self.template[0]] +
	#		         [self.tokenizer.mask_token_id] +
	#		         [-100 * self.template[1]] +
	#		         indexed_tokens[masked_index:-1] +
	#		         (-100 * self.template[2] if self.template[
	#			                                     2] > 0 else self.tokenizer.convert_tokens_to_ids(['.']))
	#		         + [self.tokenizer.sep_token_id]]
	#		# print("querys:")
	#		# print(query)
	#		# tensor = torch.Tensor(query)
	#		return query
	def get_labels(self, x_h, labels, batch):
		bz = batch
		seq = x_h.cpu().numpy().tolist()
		labels = labels.cpu().numpy().tolist()
		# print(seq)
		mask_token = 103
		template = (3, 3, 3)
		#		masked_index = seq.index(mask_token)
		queries = []
		seq_token = 102
		for i in range(bz):
			masked_index = seq[i].index(mask_token)
			seq_index = seq[i].index(seq_token)
			query = [
				labels[i][0:masked_index] +
				[-100] * template[0] +
				[labels[i][masked_index]] +
				[-100] * template[1] +
				labels[i][masked_index + 1: seq_index] +
				[-100] * template[2] +
				[labels[i][seq_index]] +
				labels[i][seq_index + 1:]
			]
			# print(query)
			queries.append(query)

		return queries

	def forward(self, events, labels, batch):
		# print(labels)
		bz = batch
		temp_events = events
		# print(events)
		prompt_tokens = [self.pseudo_token_id]
		#		seq_labels = labels
		#		# x_temp = x_hs
		#		queries = torch.Tensor(self.get_query(events, prompt_tokens, x_t=labels))

		events = torch.LongTensor(self.get_query(events, prompt_tokens, batch))
		events = events.squeeze(1).to(self.device)
		# print(events.shape)
		# labels = self.get_labels().to(device)
		# print(labels.shape)
		inputs_embeds = self.embed_input(events, batch)
		# print(inputs_embeds.shape)
		labels = torch.LongTensor(self.get_labels(temp_events, labels, batch)).to(self.device)
		labels = labels.squeeze(1).to(self.device)
		# print(labels.shape)
		# labels = self.get_labels(labels)

		outputs = self.model(inputs_embeds=inputs_embeds, labels=labels)

		mask_token = 103
		hit = 0

		for i in range(bz):
			seq = events[i].cpu().numpy().tolist()
			masked_index = seq.index(mask_token)

			# print(masked_index)
			prompt = labels[i][masked_index]
			pred_id = torch.argsort(outputs.logits[i], dim=1, descending=True)
			if pred_id[masked_index][0] == prompt:
				hit += 1
		masked_index = 103

		# print(outputs.logits.shape)
		return outputs.loss, outputs.logits, hit
		# print(queries.shape)

		#		print("queries:")
		#		print(queries)

		# print(queries)
		# queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)
		# print(queries.shape)
		# print("queries:")
		# print(queries)
		# print("x_t:")
		# print(x_ts)
		# construct label ids
		# label_id =  torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts))
		# print(label_id.shape)
		# label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape(
		# (bz, -1)).to(self.device)
		# label_ids = torch.Tensor(self.tokenizer.convert_tokens_to_ids(x_ts)).to(self.device)
		# tokenized_text2 = self.tokenizer.tokenize(x_ts)
		# label_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokenized_text2)).to(self.device)
		# print('label_id:')
		# print(label_ids)
		# attention_mask = queries != self.pad_token_id

		# get embedded input
		queries = queries.to(self.device)
		inputs_embeds = self.embed_input(queries)
		# print(inputs_embeds)
		# print(inputs_embeds1.shape)
		# print(inputs_embeds1)

		# print(x_hs)
		# xs = self.get_labels(x_hs)
		#		tokenized_text = self.tokenizer.tokenize(events)
		tokens = ['[CLS]'] + tokenizer.tokenize(events) + ['[SEP]']

		# print(tokenized_text)
		masked_index = tokens.index(labels)
		label_tokens_id = self.tokenizer.convert_tokens_to_ids(tokens)

		#		query = [
		#		         tokens[0:masked_index] +
		#		         [-100] * self.template[0] +
		#		         [tokens[masked_index]] +
		#		         [-100] * self.template[1] +
		#		         indexed_tokens[masked_index + 1:-1] +
		#		         [-100] * self.template[2] +
		#		         [self.tokenizer.sep_token_id]
		#		        ]

		# print("test_model:")
		# print(labels)
		# print(tokenized_text)
		# print(masked_index)
		# print(masked_index)
		# pred_token = self.tokenizer.tokenize('before')
		# print(masked_index)
		# indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

		# print(indexed_tokens)
		# print(len(indexed_tokens))
		# tokenized_text()
		#		query = [
		#		         indexed_tokens[0:masked_index] +
		#		         [-100] * self.template[0] +
		#		         [indexed_tokens[masked_index]] +
		#		         [-100] * self.template[1] +
		#		         indexed_tokens[masked_index + 1:] +
		#		         [-100] * 3 +
		#		         [self.tokenizer.sep_token_id]]
		query_labels = torch.Tensor(label_tokens_id).to(self.device)
		# query_labels = pad_sequence(query_labels, True, padding_value=self.pad_token_id).long().to(self.device)
		# print("query_labels:")
		# print(query_labels)
		#		print("labels:")
		#		print(labels)
		# batch ,
		# INPUT:A [MASK] B
		# LABEL A C B
		outputs = self.model(inputs_embeds=inputs_embeds, labels=query_labels)
		# print(outputs.loss)
		# print(outputs.logits)
		# print(outputs.logits.shape)
		pred_ids = torch.argsort(outputs.logits, dim=2, descending=True)

		text1 = 'before'
		tokenized_text1 = self.tokenizer.tokenize(text1)
		indexed_tokens_before = self.tokenizer.convert_tokens_to_ids(tokenized_text1)
		#		print("before token:")
		#		print(indexed_tokens_before)

		text2 = 'after'
		tokenized_text2 = self.tokenizer.tokenize(text2)
		indexed_tokens_after = self.tokenizer.convert_tokens_to_ids(tokenized_text2)
		#		print("after token:")
		#		print(indexed_tokens_after)

		#		pred_logits = outputs.logits
		##
		##		print(pred_logits.shape)
		#		tokenized_text = self.tokenizer.tokenize(events)
		#		masked_index = tokenized_text.index(labels)
		#		print("masked_index:")
		#		print(masked_index)
		#		print(len(tokenized_text))
		#		print(indexed_tokens_before)
		#		print(indexed_tokens_after)
		#		if seq_labels == 'before':
		#			print("before")
		#			if pred_logits[0][masked_index + 3][indexed_tokens_before] >= pred_logits[0][masked_index + 3][indexed_tokens_after]:
		#				hit = 1
		#			else:
		#				hit = 0
		#
		#		elif seq_labels == 'after':
		#			print("after")
		#			if pred_logits[0][masked_index + 3][indexed_tokens_before] < pred_logits[0][masked_index + 3][indexed_tokens_after]:
		#				hit = 1
		#			else:
		#				hit = 0

		#
		# print(pred_ids.shape)
		# # print(pred_ids[0, masked_index, 0])
		# # print(pred_ids[0,masked_index,:][:20])
		# # print()
		# print(indexed_tokens[masked_index])
		# print(pred_ids[0, masked_index + 3, 0])
		print(indexed_tokens_before)
		print(indexed_tokens_after)
		print(indexed_tokens[masked_index])

		if indexed_tokens[masked_index] == pred_ids[0, masked_index + 3, 0]:
			hit = 1
		else:
			hit = 0
		return outputs.loss, hit, outputs.logits