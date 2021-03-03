
from collections import Counter
from operator import itemgetter

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_data(filename, has_label=False, pure_ids=False, verbose=False):
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
        if pure_ids:
            seq_lengths.append(len(text_a.split(" ")) + len(text_b.split(" ")) + 3)
        else:
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


def get_train_valid(texts, labels, test_size=0.05, random_state=2021):
    x_train, x_valid, y_train, y_valid = train_test_split(
        texts, labels, stratify=labels, test_size=test_size, random_state=random_state)
    return x_train, x_valid, y_train, y_valid


def get_data(data, data_index):
    return [data[i] for i in data_index]


def yield_train_valid(texts, labels, nfolds=5, random_state=2021):
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    for train_index, valid_index in kf.split(texts, labels):
        x_train, y_train = get_data(texts, train_index), get_data(labels, train_index)
        x_valid, y_valid = get_data(texts, valid_index), get_data(labels, valid_index)
        yield (x_train, x_valid, y_train, y_valid)


def auc_score(y_trues, y_preds):
    y_trues = list(map(int, y_trues))
    match_cnt = sum(y_trues)
    mismatch_cnt = len(y_trues) - match_cnt
    rank_scores = [itemgetter(0)(t) for t in sorted(enumerate(y_preds, 1), key=itemgetter(1))]
    pos_rank_scores = [rank for rank, label in zip(rank_scores, y_trues) if label == 1]
    return (sum(pos_rank_scores) - (1 + match_cnt) / 2) / (match_cnt * mismatch_cnt)
