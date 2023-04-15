from typing import Union, Iterable
import numpy as np
from . import rouge
from nltk.translate.bleu_score import sentence_bleu

##########################
# Rouge
##########################

def events_to_sentence(events: Iterable[str], delimiter: str=''):
    """
        Connect events to sentence for evaluation.
    """
    return delimiter.join(events)

def rouge_l(pred: Union[Iterable[str], str], ref: Union[Iterable[str], str]) -> dict:
    """
        Calculates Rouge-L score for a pair of predicted and reference sequences.
    """

    score = rouge.rouge_l_sentence_level(pred, ref)
    return {'recall':score.recall, 'precision':score.precision, 'f1':score.f1_measure}

def rouge_n(pred: Union[Iterable[str], str], ref: Union[Iterable[str], str], ngram: int) -> dict:
    """
        Calculates Rouge-n score for a pair of predicted and reference sequences.
    """   
        
    score = rouge.rouge_n_sentence_level(pred, ref, ngram)
    return {'recall':score.recall, 'precision':score.precision, 'f1':score.f1_measure}

def rouge_w(pred: Union[Iterable[str], str], ref: Union[Iterable[str], str], weight:float) -> dict:
    """
        Calculates Rouge-n score for a pair of predicted and reference sequences.
    """  
        
    score = rouge.rouge_w_sentence_level(pred, ref, weight)
    return {'recall':score.recall, 'precision':score.precision, 'f1':score.f1_measure}           

def _str_level(s):
    try:
        # string is given
        if isinstance(s, str):
            return -1
        # list of strings is given
        elif isinstance(s[0], str):
            return 1
        # list of list of strings is given
        elif isinstance(s[0][0], str):
            return 2
        # list of list of of list strings is given
        elif isinstance(s[0][0][0], str):
            return 3
        else:
            return -1
    except:
        return -1
        
def _check_input(preds, refs):
    pl = _str_level(preds)
    rl = _str_level(refs)

    assert pl != -1 and rl != -1, 'Either preds or refs are invalid'
    assert rl - pl == 1, 'Dimension of refs is invalid'
    
    return pl, rl

def _append_dict(dict_list):
    buf = {}
    for d in dict_list:
        for k, v in d.items():
            if k not in buf: buf[k] = []
            buf[k].append(v)
    return buf
    
def _rouge_topk(pred: Union[Iterable[Iterable[str]], Iterable[str]],
                ref: Union[Iterable[Iterable[Iterable[str]]], Iterable[Iterable[str]]],
                topk: int,
                metric: str,
                rouge_fn,
                return_index: bool=False,
                *rouge_args, 
                **rouge_kwargs):
    """
        Finds the top k candidates from reference seqs based on certain metric.
    """
    def _find_topk(p, rs):
        scores = []
        for i, r in enumerate(rs):
            score = rouge_fn(p, r, *rouge_args, **rouge_kwargs)
            if return_index:
                score.update({'index': i})
            scores.append(score)
        topk_ref = sorted(scores, key=lambda r: r[metric], reverse=True)
        if topk != -1:
            topk_ref = topk_ref[:topk]
        
        return _append_dict(topk_ref)
    
    pl, _ = _check_input(pred, ref)

    # single input
    if pl == 1:
        pred = [pred]

    ret = []
    for i in range(len(pred)):
        ret.append(_find_topk(pred[i], ref[i]))
    ret = _append_dict(ret)

    return ret

def rouge_l_topk(pred, ref, topk, metric, return_index=False) -> dict:
    """
        Calculates Rouge-L for predicted sequence and the top K closest reference sequences
        :param pred: (multiple) predicted sequences of words
        :param ref: reference sequences of words
        :param topk: number of closest reference sequences to return
        :param metric: name of metric for evaluation of reference sequences
        :param return_index: return index of topk reference sequences, default=False
    """
    return _rouge_topk(pred, ref, topk, metric, rouge_l, return_index=return_index)

def rouge_n_topk(pred, ref, topk, metric, ngram, return_index=False) -> dict:
    """
        Calculates Rouge-N for predicted sequence and the top K closest reference sequences
        :param pred: (multiple) predicted sequences of words
        :param ref: reference sequences of words
        :param topk: number of closest reference sequences to return
        :param metric: name of metric for evaluation of reference sequences
        :param ngram: n-gram to evaluate
        :param return_index: return index of topk reference sequences, default=False
    """
    return _rouge_topk(pred, ref, topk, metric, rouge_n, return_index=return_index, ngram=ngram)

def rouge_w_topk(pred, ref, topk, metric, weight=None, return_index=False) -> dict:
    """
        Calculates Rouge-W for predicted sequence and the top K closest reference sequences
        :param pred: (multiple) predicted sequences of words; each sequence must be Iterable[str]
        :param ref: reference sequences of words; each sequence must be Iterable[str]
        :param topk: number of closest reference sequences to return
        :param metric: name of metric for evaluation of reference sequences
        :param weight: the weight factor passed to the weight function, default=None means weight=1.2
        :param return_index: return index of topk reference sequences, default=False
    """
    return _rouge_topk(pred, ref, topk, metric, rouge_w, return_index=return_index, weight=weight)

##########################
# BLEU
##########################

def bleu(pred: Union[Iterable[str], str], 
         ref: Union[Iterable[str], str],
         ngram: int=1, 
         weights=None,
         smoothing_function=None,
         auto_reweigh=False,) -> float:

    assert weights is None or ngram == len(weights), f'ngram={ngram} is not equal to length of weights={len(weights)}'
    
    if isinstance(pred, str):
        pred = pred.split() # 'abc' -> ['a', 'b', 'c']
    if isinstance(ref, str):
        ref = ref.split()
    if weights is None:
        weights = tuple([1./float(ngram) for _ in range(ngram)])

    return sentence_bleu(ref, pred, weights, smoothing_function, auto_reweigh)

def bleu_topk(pred: Union[Iterable[Iterable[str]], Iterable[str]],
              ref: Union[Iterable[Iterable[Iterable[str]]], Iterable[Iterable[str]]],
              topk: int,
              return_index: bool=False,
              ngram: int=1,
              weights=None,
              smoothing_function=None,
              auto_reweigh=False) -> dict:
    """
        Calculates BLEU for predicted sequence and the top K closest reference sequences
        :param pred: (multiple) predicted sequences of words; each sequence must be Iterable[str]
        :param ref: reference sequences of words; each sequence must be Iterable[str]
        :param topk: number of closest reference sequences to return
        :param ngram: n-gram to evalute for, default=1
        :param return_index: return index of topk reference sequences, default=False
    """
    assert weights is None or ngram == len(weights), f'ngram={ngram} is not equal to length of weights={len(weights)}'

    pl, _ = _check_input(pred, ref)

    # single input
    if pl == 1:
        pred = [pred]
    
    ret = []
    for p in pred:
        scores = []
        for r in ref:
            scores.append(bleu(p, r, ngram, weights, smoothing_function, auto_reweigh))
        
        if return_index:
            scores = sorted(list(zip(np.arange(len(scores), dtype=int), scores)), 
                            key=lambda s: s[1], reverse=True)
            scores = scores[:topk] if topk != -1 else scores
            ret.append({'bleu': [s[1] for s in scores], 'index': [s[0] for s in scores]})
        else:
            scores = sorted(scores, reverse=True)
            scores = scores[:topk] if topk != -1 else scores
            ret.append({'bleu': scores})
    
    ret = _append_dict(ret)
    return ret
    
    
            
        
    