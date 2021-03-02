
from collections import Counter
from operator import itemgetter

import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filename, has_label=False, verbose=False, pure_id=False):
    texts, labels = [], []
    seq_lengths = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        if has_label:
            text_a, text_b, label = line.split("\t", 2)
            labels.append(label)
        else:
            text_a, text_b = line.split("\t", 1)
        if pure_id:
            text_a = text_a.split(" ")
            text_b = text_b.split(" ")
        texts.append([text_a, text_b])
        seq_lengths.append(len(text_a) + len(text_b) + 3)

    seq_len_50 = int(np.mean(seq_lengths))
    seq_len_90 = np.percentile(seq_lengths, 90)
    seq_len_99 = np.percentile(seq_lengths, 99)
    if verbose:
        print(f"序列长度 50%/90%/99%: {seq_len_50}/{seq_len_90}/{seq_len_99}")

    if has_label:
        if verbose:
            print("标签分布比例: ", dict(Counter(labels)))
        return texts, labels
    else:
        return texts


def get_train_valid(texts, labels, test_size=0.05):
    x_train, x_valid, y_train, y_valid = train_test_split(
        texts, labels, stratify=labels, test_size=test_size, random_state=2021)
    return x_train, x_valid, y_train, y_valid


def auc_score(y_trues, y_preds):
    y_trues = list(map(int, y_trues))
    match_cnt = sum(y_trues)
    mismatch_cnt = len(y_trues) - match_cnt
    rank_scores = [itemgetter(0)(t) for t in sorted(enumerate(y_preds, 1), key=itemgetter(1))]
    pos_rank_scores = [rank for rank, label in zip(rank_scores, y_trues) if label == 1]
    return (sum(pos_rank_scores) - (1 + match_cnt) / 2) / (match_cnt * mismatch_cnt)
