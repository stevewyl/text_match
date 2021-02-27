import json
import sys

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import sequence_padding
from keras.models import Model

# 加载kernel和bias向量
model_version = f"2021-{sys.argv[1]}"
model_dir = "../user_data/model_data/bert_whitening"
kernel = np.load(f"{model_dir}-{model_version}/kernel.npy")
bias = np.load(f"{model_dir}-{model_version}/bias.npy")
config = json.load(open(f"{model_dir}-{model_version}/config.json"))


def load_data(filename):
    texts = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        text_a, text_b = line.split("\t", 1)
        texts.append([text_a, text_b])
    return texts


def convert_to_vecs(encoder, data, maxlen=64):
    """转换文本数据为id形式"""
    a_token_ids, b_token_ids = [], []
    for d in data:
        token_ids = tokenizer.encode(d[0], maxlen=maxlen)[0]
        a_token_ids.append(token_ids)
        token_ids = tokenizer.encode(d[1], maxlen=maxlen)[0]
        b_token_ids.append(token_ids)
    a_token_ids = sequence_padding(a_token_ids)
    b_token_ids = sequence_padding(b_token_ids)
    a_vecs = encoder.predict([a_token_ids,
                              np.zeros_like(a_token_ids)],
                             verbose=True)
    b_vecs = encoder.predict([b_token_ids,
                              np.zeros_like(b_token_ids)],
                             verbose=True)
    return a_vecs, b_vecs


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs ** 2).sum(axis=1, keepdims=True) ** 0.5


class GlobalAveragePooling1D(keras.layers.GlobalAveragePooling1D):
    """自定义全局池化
    """
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())[:, :, None]
            return K.sum(inputs * mask, axis=1) / K.sum(mask, axis=1)
        else:
            return K.mean(inputs, axis=1)


def get_encoder(config_path, checkpoint_path, n_last=2):
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
    return encoder


# 读取数据
texts = load_data("../tcdata/oppo_breeno_round1_data/testA.tsv")
# 建立分词器
tokenizer = Tokenizer(config["dict_path"], do_lower_case=True)
# 构建模型
encoder = get_encoder(config["config_path"], config["checkpoint_path"], config["last_layer"])
# 向量化
a_vecs, b_vecs = convert_to_vecs(encoder, texts, 40)
# 变换，标准化，相似度
a_vecs = transform_and_normalize(a_vecs, kernel, bias)
b_vecs = transform_and_normalize(b_vecs, kernel, bias)
sims = (a_vecs * b_vecs).sum(axis=1)
# 保存结果
with open("../prediction_result/result_whiten.txt", "w") as fw:
    for score in sims:
        fw.write(str(score) + "\n")
