from sklearn.metrics import precision_recall_curve, average_precision_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torch

class Metric():
    def __init__(self, keys, best_key=None):
        self.vals = dict(zip(keys, [None for _ in keys]))
        self.best_key = best_key

    def update(self, vals):
        self.vals.update(vals)

    def best(self):
        if self.best_key is None:
            return None
        return self.vals[self.best_key]

    def __call__(self):
        return self.vals

def buffer(buf, x):
    if buf is None:
        buf = x
    else:
        if isinstance(buf, torch.Tensor):
            buf = torch.cat([buf, x], dim=0)
        elif isinstance(buf, np.ndarray):
            buf = np.cat([buf, x], dim=0)
        else:
            buf.extend(x)
    return buf

def bi_ap(y_true, score):
    if len(score.shape) > 1:
        score = score[:, 1]
    return average_precision_score(y_true, score)

def bi_acc(y_true, score, thresh=0.5, subset_label=None):
    if subset_label is not None:
        y_true = y_true[y_true==subset_label]
        score = score[y_true==subset_label]
    if len(score.shape) > 1:
        score = score[:, 1]
    y_pred = score > thresh
    return accuracy_score(y_true, y_pred)

def bi_fp(y_true, score, thresh=0.5):
    if len(score.shape) > 1:
        score = score[:, 1]
    y_pred = score > thresh
    return sum(y_pred[y_true!=1])

def bi_tp(y_true, score, thresh=0.5):
    if len(score.shape) > 1:
        score = score[:, 1]
    y_pred = score > thresh
    return sum(y_pred[y_true==1])

def bi_fp_tp_ratio(y_true, score, thresh=0.5):
    if len(score.shape) > 1:
        score = score[:, 1]
    y_pred = score > thresh
    fp = bi_fp(y_true, score, thresh)
    tp = bi_tp(y_true, score, thresh)
    if tp == 0:
        return -1
    return fp / tp

def bi_total_p(y_true, score, thresh=0.5):
    if len(score.shape) > 1:
        score = score[:, 1]
    y_pred = score > thresh
    return sum(y_pred)

def find_thresh(y_true, score, tar={'recall': 0.8}):
    if len(score.shape) > 1:
        score = score[:, 1]
    tar, tar_val = list(tar.keys())[0], list(tar.values())[0]
    if tar in {'recall', 'precision'}:
        ps, rs, ths = precision_recall_curve(y_true, score)
        if tar == 'recall':
            idx = np.argmin(np.abs(rs - tar_val))
            return ths[idx]
        if tar == 'precision':
            idx = np.argmin(np.abs(ps - tar_val))
            return ths[idx]

def bi_prc(y_true, score, display=False, save_path=None, ax=None):
    if len(score.shape) > 1:
        score = score[:, 1]    
    ps, rs, ths = precision_recall_curve(y_true, score)

    fig = None
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(rs, ps, label='1')
    ax.set_xlim((-0.05, 1.05)), ax.set_ylim((-0.05, 1.05))
    ax.set_xlabel('Recall'), ax.set_ylabel('Precision')
    ax.set_xticks(ticks=np.arange(0, 1.1, 0.1)), ax.set_yticks(ticks=np.arange(0, 1.1, 0.1))
    ax.grid(linestyle='dashed'), ax.legend(loc='lower left')
    ax.set_title(f'Precision-Recall Curve')
    if save_path is not None:
        plt.savefig(f"{save_path}")
    if display:
        plt.show()
    if fig is not None:
        plt.close(fig)
    return ax

def eval_acc_target(acc_train, acc_test):
    if acc_train < 0.8:
        return -1 + acc_test
    else:
        return acc_test 

def eval_fptp_target(fptp_train, fptp_test):
    if fptp_train >= 2.5:
        return 10 + fptp_test
    else:
        return fptp_test