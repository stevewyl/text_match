"""
BERT Whitening
Refer from: https://kexue.fm/archives/8069
"""

import sys

import numpy as np
import scipy.stats
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import open, sequence_padding
from keras.models import Model
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import load_data, get_train_valid, auc_score


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
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    return a_vecs, b_vecs, np.array(labels)


def compute_kernel_bias(vecs):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    """
    vecs = np.concatenate(vecs, axis=0)
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W[:, :256], -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


# bert配置
model_name = sys.argv[1]
env = sys.argv[2] or "home"
model_full_name = {
    "roberta": "chinese_roberta_wwm_ext_L-12_H-768_A-12",
    "macbert": "chinese_macbert_base"
}
env_base_dir = {
    "home": "/media/stevewyl/Extreme SSD/work",
    "jd": "/home/wyl/disk0"
}
# 根据你模型的存放位置进行修改
model_dir = f"{env_base_dir[env]}/bert_models/{model_full_name[model_name]}"
config_path = f"{model_dir}/bert_config.json"
checkpoint_path =  f"{model_dir}/bert_model.ckpt"
dict_path =  f"{model_dir}/vocab.txt"

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

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

# 取倒数两层的结果
n_last = 1
if n_last > 1:
    outputs = []
    for i in range(n_last):
        outputs.append(GlobalAveragePooling1D()(encoder_layers[-i]))
    output = keras.layers.Average()(outputs)
else:
    output = GlobalAveragePooling1D()(encoder_layers[-1])

# 最后的编码器
encoder = Model(bert.inputs, output)
# print(encoder.summary())

# 读取数据
texts, labels = load_data("../../tcdata/oppo_breeno_round1_data/train.tsv", has_label=True)
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

# 计算变换矩阵和偏置项
kernel, bias = compute_kernel_bias([v for vecs in all_vecs for v in vecs])
# TODO: 保存kernel和bias向量


# 变换，标准化，相似度
sim_scores = []
for name, (a_vecs, b_vecs), labels in zip(all_names, all_vecs, all_labels):
    a_vecs = transform_and_normalize(a_vecs, None, bias)
    b_vecs = transform_and_normalize(b_vecs, None, bias)
    sims = (a_vecs * b_vecs).sum(axis=1)
    y_preds = [1 if score > 0.5 else 0 for score in sims]
    f1 = f1_score(labels, y_preds) * 100
    precision = precision_score(labels, y_preds) * 100
    recall = recall_score(labels, y_preds) * 100
    auc = auc_score(labels, sims) * 100
    print(f"[{name}] F1/Precison/Recall/Auc: {f1:.2f}/{precision:.2f}/{recall:.2f}/{auc:.2f}")
