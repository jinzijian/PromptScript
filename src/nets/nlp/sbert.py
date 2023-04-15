from sentence_transformers import SentenceTransformer, models
import torch
import torch.nn as nn
# import .common

def SbertEncoder(model_name, max_seq_length=512):
    return models.Transformer(model_name, max_seq_length=max_seq_length)

def SbertPooling(encoder=None, dim=None):
    assert encoder is not None or dim is not None, 'Either encoder or dim should be given.'
    if encoder is not None:
        dim = encoder.get_word_embedding_dimension()
    return models.Pooling(dim)

def SbertHead(pooling=None, in_dim=None, out_dim=128, act=None):
    assert pooling is not None or in_dim is not None, 'Either pooling layer or in_dim should be given'
    if in_dim is None:
        in_dim = pooling.get_sentence_embedding_dimension()
    return models.Dense(in_features=in_dim, out_features=out_dim, activation_function=act)

def SBERT(model_name, device='cpu', max_seq_length=512, use_head=False, out_dim=128, act=nn.Tanh()):
    encoder = SbertEncoder(model_name, max_seq_length)
    pooling = SbertPooling(encoder)
    if use_head:
        head = SbertHead(pooling, out_dim=out_dim, act=act)
        return SentenceTransformer(modules=[encoder, pooling, head])
    return SentenceTransformer(modules=[encoder, pooling], device=device)