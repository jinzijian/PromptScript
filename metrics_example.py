import numpy as np
from metrics import *

# preds = [['send to a', 'receive by b'], ['give to c', 'gather by d']]
# refs = [
#     [['send to a', 'receive by b'], ['give to c', 'gather by d']], 
#     [['send to a', 'receive by b'], ['give to c', 'gather by d'], ['mail to e', 'retrieve by f']]
# ]

preds = [['A','B','C','D','E'], ['G', 'K']]
refs = [
    [['A', 'B', 'C', 'E', 'D'], ['A', 'E', 'C', 'D', 'B']], 
    [['G'], ['K', 'G', 'K']]
]

# rouge - pair-wise
print(rouge_l(refs[0][1], preds[0]))
print(rouge_l(preds[0], refs[0][0]))
print(rouge_n(preds[0], refs[0][0], 1))
print(rouge_n(preds[0], refs[0][0], 2))

# rouge - topk - single input
print(rouge_l_topk(preds[0], refs[0], topk=1, metric='f1', return_index=False))
print(rouge_l_topk(preds[0], refs[1], topk=1, metric='recall', return_index=True))
print(rouge_n_topk(preds[0], refs[0], topk=2, metric='precision', ngram=2, return_index=True))
print(rouge_w_topk(preds[0], refs[1], topk=-1, metric='f1', return_index=False))

# rouge - topk - multiple input
print(rouge_l_topk(preds, refs, topk=1, metric='f1', return_index=False))
print(rouge_l_topk(preds, refs, topk=1, metric='recall', return_index=True))
print(rouge_n_topk(preds, refs, topk=2, metric='precision', ngram=1, return_index=True))
print(rouge_n_topk(preds, refs, topk=-1, metric='f1', ngram=2, return_index=False))
print(rouge_w_topk(preds, refs, topk=-1, metric='f1', return_index=True))

# bleu - pairwise
print(bleu(preds[0], refs[0][0]))

# bleu - topk - single input
print(bleu_topk(preds[0], refs[1], topk=1, return_index=True))
print(bleu_topk(preds[0], refs[0], topk=-1))

# bleu - topk - multiple input
print(bleu_topk(preds, refs, topk=1))
print(bleu_topk(preds, refs, topk=-1, return_index=True))
