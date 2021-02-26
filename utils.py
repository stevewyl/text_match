
from collections import Counter
from operator import itemgetter

import numpy as np
from sklearn.model_selection import train_test_split


def load_data(filename, has_label=False):
    texts, labels = [], []
    seq_lengths = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        if has_label:
            text_a, text_b, label = line.split("\t", 2)
            labels.append(label)
        else:
            text_a, text_b = line.split("\t", 1)
        texts.append([text_a, text_b])
        seq_lengths.append(len(text_a) + len(text_b) + 3)

    seq_len_50 = int(np.mean(seq_lengths))
    seq_len_90 = np.percentile(seq_lengths, 90)
    seq_len_99 = np.percentile(seq_lengths, 99)
    print(f"序列长度 50%/90%/99%: {seq_len_50}/{seq_len_90}/{seq_len_99}")

    if has_label:
        print("标签分布比例: ", dict(Counter(labels)))
        return texts, labels
    else:
        return texts


def get_train_valid(texts, labels, test_size=0.05):
    x_train, x_valid, y_train, y_valid = train_test_split(
        texts, labels, stratify=labels, test_size=test_size)
    return x_train, x_valid, y_train, y_valid


def auc_score(y_trues, y_preds):
    y_trues = list(map(int, y_trues))
    match_cnt = sum(y_trues)
    mismatch_cnt = len(y_trues) - match_cnt
    rank_scores = [itemgetter(0)(t) for t in sorted(enumerate(y_preds, 1), key=itemgetter(1))]
    rank_scores = [score - match_cnt * (1 + match_cnt) / 2 for score in rank_scores]
    return sum(rank_scores) / (match_cnt * mismatch_cnt)
