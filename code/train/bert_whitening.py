"""
BERT Whitening
Refer from: https://kexue.fm/archives/8069
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import scipy.stats
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from keras.models import Model
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from utils import load_data, get_train_valid

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


def convert_to_vecs(encoder, data, maxlen=64):
    """转换文本数据为id形式"""
    a_token_ids, b_token_ids, labels = [], [], []
    for d in data:
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
        labels.append(d[2])
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             batch_size=1024,
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             batch_size=1024,
                             verbose=True)
    return a_vecs, b_vecs, np.array(labels)


def compute_kernel_bias(vecs, dim=-1):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :dim], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


def get_encoder(config_path, checkpoint_path, n_last=2, verbose=False):
    # 建立模型
    bert = build_transformer_model(config_path, checkpoint_path)

    # 获取每一层transformer的输出
    encoder_layers, count = [], 0
    while True:
        try:
            output = bert.get_layer(
                'Transformer-%d-FeedForward-Norm' % count
            ).output
            encoder_layers.append(output)
            count += 1
        except:
            break

    if n_last > 1:
        outputs = []
        for i in range(n_last):
            outputs.append(GlobalAveragePooling1D()(encoder_layers[-i]))
        output = keras.layers.Average()(outputs)
    else:
        output = GlobalAveragePooling1D()(encoder_layers[-1])

    # 最后的编码器
    encoder = Model(bert.inputs, output)
    if verbose:
        print(encoder.summary())
    return encoder


arg_parser = argparse.ArgumentParser(description="Unsupervised Text Similarity (Whitening)")
arg_parser.add_argument("-e", "--env", type=str, default="home", help="script running environment")
arg_parser.add_argument("-l", "--last_layer", type=int, default=2, help="number of last Transformer layers as output")
arg_parser.add_argument("-w", "--whiten", action="store_true", help="whether to use whitening operation")
arg_parser.add_argument("-p", "--plm", type=str, default="roberta", help="pretrained language model name")
arg_parser.add_argument("-d", "--dim", type=int, default=256, help="First dim dimensions of the output vector")
args = arg_parser.parse_args()

# bert配置
model_full_name = {
    "roberta": "chinese_roberta_wwm_ext_L-12_H-768_A-12",
    "macbert": "chinese_macbert_base"
}
env_base_dir = {
    "home": "/media/stevewyl/Extreme SSD/work",
    "jd": "/home/wyl/disk0"
}
# 根据你模型的存放位置进行修改
model_dir = f"{env_base_dir[args.env]}/bert_models/{model_full_name[args.plm]}"
config_path = f"{model_dir}/bert_config.json"
checkpoint_path =  f"{model_dir}/bert_model.ckpt"
dict_path =  f"{model_dir}/vocab.txt"

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# 构建模型
encoder = get_encoder(config_path, checkpoint_path, args.last_layer)

# 读取数据
texts, labels = load_data("../tcdata/oppo_breeno_round1_data/train.tsv", has_label=True)
x_train, x_valid, y_train, y_valid = get_train_valid(texts, labels)
datasets = {
    "train": [x + [int(y)] for x, y in zip(x_train, y_train)],
    "dev": [x + [int(y)] for x, y in zip(x_valid, y_valid)]
}

# 语料向量化
all_names, all_vecs, all_labels = [], [], []
for name, data in datasets.items():
    a_vecs, b_vecs, labels = convert_to_vecs(encoder, data, 40)
    all_names.append(name)
    all_vecs.append((a_vecs, b_vecs))
    all_labels.append(labels)

if args.whiten:
    # 计算变换矩阵和偏置项
    kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs], args.dim)
    # 保存kernel和bias向量
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"../user_data/model_data/bert_whitening-{args.plm}-{timestamp}"
    os.makedirs(save_path, exist_ok=True)
    np.save(save_path + "/kernel.npy", kernel)
    np.save(save_path + "/bias.npy", bias)
    config = {
        "last_layer": args.last_layer,
        "plm": args.plm,
        "dim": args.dim,
        "checkpoint_path": checkpoint_path,
        "dict_path": dict_path,
        "config_path": config_path
    }
    with open(f"{save_path}/config.json", "w") as fw:
        json.dump(config, fw, indent=2)
else:
    kernel = None
    bias = None
    save_path = None

# 变换，标准化，相似度
results = []
for name, (a_vecs, b_vecs), labels in zip(all_names, all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, kernel, bias)
    b_vecs = transform_and_normalize(b_vecs, kernel, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)
    y_preds = [1 if score > 0.5 else 0 for score in sims]
    f1 = f1_score(labels, y_preds) * 100
    precision = precision_score(labels, y_preds) * 100
    recall = recall_score(labels, y_preds) * 100
    auc = roc_auc_score(labels, sims) * 100
    print(f"[{name}] F1/Precison/Recall/Auc: {f1:.2f}/{precision:.2f}/{recall:.2f}/{auc:.2f}")
    results.append([name, f1, precision, recall, auc])

if save_path:
    result_fn = f"{save_path}/result.csv"
    with open(result_fn, "w") as fw:
        fw.write(",".join(["dataset_name", "f1", "precision", "recall", "auc"]) + "\n")
        for res in results:
            fw.write(",".join([str(v) if type(v) == np.float64 else v for v in res]) + "\n")
