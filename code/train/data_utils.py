import random
from collections import defaultdict

import jellyfish

from utils import load_data


def edit_distance(sent1, sent2):
    vocab = set()
    sent1_tokens = set(sent1.split(" "))
    sent2_tokens = set(sent2.split(" "))
    vocab.update(sent1_tokens)
    vocab.update(sent2_tokens)
    new_vocab = {v: chr(idx + 97) for idx, v in enumerate(list(vocab))}
    new_sent1 = "".join([new_vocab[t] for t in sent1.split(" ")])
    new_sent2 = "".join([new_vocab[t] for t in sent2.split(" ")])
    return jellyfish.levenshtein_distance(new_sent1, new_sent2)

data_fn = "train_id.tsv"
base_dir = "../tcdata/oppo_breeno_round1_data"
texts, labels = load_data(f"{base_dir}/{data_fn}", has_label=True)

sent2id = {}
idx = 0
for sent1, sent2 in texts:
    if sent1 not in sent2id:
        sent2id[sent1] = idx
        idx += 1
    if sent2 not in sent2id:
        sent2id[sent2] = idx
        idx += 1
id2sent = {idx: sent for sent, idx in sent2id.items()}

similar_sents = defaultdict(set)
exist_pairs = set()
edit_scores = {}
for (sent1, sent2), label in zip(texts, labels):
    if label == "0":
        continue
    edit_score = edit_distance(sent1, sent2)
    sent1 = sent2id[sent1]
    sent2 = sent2id[sent2]
    pair1 = f"{sent1}-{sent2}"
    pair2 = f"{sent2}-{sent1}"
    exist_pairs.add(pair1)
    exist_pairs.add(pair2)
    similar_sents[sent1].add(sent2)
    similar_sents[sent2].add(sent1)
    edit_scores[pair1] = edit_score
    edit_scores[pair2] = edit_score

similar_sents = {sent: list(sents_set) for sent, sents_set in similar_sents.items() if len(sents_set) > 1}

new_pairs = []
for sid, sid_list in similar_sents.items():
    ori_sent = id2sent[sid]
    for i in range(len(sid_list) - 1):
        ori_pair1 = f"{sid}-{sid_list[i]}"
        ori_score = edit_scores[ori_pair1]
        for j in range(i + 1, len(sid_list)):
            sent1, sent2 = id2sent[sid_list[i]], id2sent[sid_list[j]]
            new_score = edit_distance(ori_sent, sent2)
            if new_score <= ori_score:
                pair1 = f"{sid_list[i]}-{sid_list[j]}"
                pair2 = f"{sid_list[j]}-{sid_list[i]}"
                if pair1 not in exist_pairs and pair2 not in exist_pairs:
                    if random.random() > 0.5:
                        new_pairs.append([sent1, sent2])
                    else:
                        new_pairs.append([sent2, sent1])
                else:
                    exist_pairs.add(pair1)
                    exist_pairs.add(pair2)

with open(f"{base_dir}/train_id_aug.tsv", "w") as fw:
    for sent1, sent2 in new_pairs:
        fw.write("\t".join([sent1, sent2, "1"]) + "\n")
