import argparse
import json
import logging
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.tokenizers import load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import truncate_sequences
from sentence_transformers import LoggingHandler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

arg_parser = argparse.ArgumentParser(description="MLM model for desensitization data")
arg_parser.add_argument("-e", "--env", type=str, default="home", help="script running environment")
arg_parser.add_argument("-p", "--plm", type=str, default="bert", help="pretrained language model name")
arg_parser.add_argument("-s", "--max_length", type=int, default=32, help="max sequence length")
arg_parser.add_argument("-b", "--batch_size", type=str, default=64, help="training batch size")
arg_parser.add_argument("--min_count", type=int, default=5, help="min count of desensitization vocab")
arg_parser.add_argument("--epoches", type=int, default=100, help="number of training epoches")
args = arg_parser.parse_args()

# 日志
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_path = f"../logs/match_mlm-{args.plm}-{timestamp}.log"
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler(), logging.FileHandler(log_path)])
logger = logging.getLogger(__name__)

# bert配置
model_full_name = {
    "roberta": "chinese_roberta_wwm_ext_L-12_H-768_A-12",
    "macbert": "chinese_macbert_base",
    "uer": "uer_bert_base",
    "bert": "chinese_L-12_H-768_A-12",
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
model_save_dir = f"../user_data/model_data/match_mlm-{args.plm}-{timestamp}"
os.makedirs(model_save_dir, exist_ok=True)


def load_data(filename):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            else:
                a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            truncate_sequences(args.max_length, -1, a, b)
            D.append((a, b, c))
    return D


# 加载数据集
data_dir = "../tcdata/oppo_breeno_round1_data"
data = load_data(f"{data_dir}/train_id.tsv")
train_data = [d for i, d in enumerate(data) if i % 10 != 0]
valid_data = [d for i, d in enumerate(data) if i % 10 == 0]
test_data = load_data(f"{data_dir}/testA_id.tsv")

# 统计词频
tokens = {}
for d in data + test_data:
    for i in d[0] + d[1]:
        tokens[i] = tokens.get(i, 0) + 1

tokens = {i: j for i, j in tokens.items() if j >= args.min_count}
tokens = sorted(tokens.items(), key=lambda s: -s[1])
tokens = {
    t[0]: i + 7
    for i, t in enumerate(tokens)
}  # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes

# BERT词频
counts = json.load(open('../user_data/tmp_data/counts.json'))
del counts['[CLS]']
del counts['[SEP]']
token_dict = load_vocab(dict_path)
freqs = [
    counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
]
keep_tokens = list(np.argsort(freqs)[::-1])

# 模拟未标注
for d in valid_data + test_data:
    train_data.append((d[0], d[1], -5))


def random_mask(text_ids):
    """随机mask
    """
    input_ids, output_ids = [], []
    rands = np.random.random(len(text_ids))
    for r, i in zip(rands, text_ids):
        if r < 0.15 * 0.8:
            input_ids.append(4)
            output_ids.append(i)
        elif r < 0.15 * 0.9:
            input_ids.append(i)
            output_ids.append(i)
        elif r < 0.15:
            input_ids.append(np.random.choice(len(tokens)) + 7)
            output_ids.append(i)
        else:
            input_ids.append(i)
            output_ids.append(0)
    return input_ids, output_ids


def sample_convert(text1, text2, label, random=False):
    """转换为MLM格式
    """
    text1_ids = [tokens.get(t, 1) for t in text1]
    text2_ids = [tokens.get(t, 1) for t in text2]
    if random:
        if np.random.random() < 0.5:
            text1_ids, text2_ids = text2_ids, text1_ids
        text1_ids, out1_ids = random_mask(text1_ids)
        text2_ids, out2_ids = random_mask(text2_ids)
    else:
        out1_ids = [0] * len(text1_ids)
        out2_ids = [0] * len(text2_ids)
    token_ids = [2] + text1_ids + [3] + text2_ids + [3]
    segment_ids = [0] * (len(text1_ids) + 2) + [1] * (len(text2_ids) + 1)
    output_ids = [label + 5] + out1_ids + [0] + out2_ids + [0]
    return token_ids, segment_ids, output_ids


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids, output_ids = sample_convert(
                text1, text2, label, random
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(output_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [batch_token_ids, batch_segment_ids], batch_output_ids
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


# 转换数据集
train_generator = data_generator(train_data, args.batch_size)
valid_generator = data_generator(valid_data, args.batch_size)
test_generator = data_generator(test_data, args.batch_size)


def masked_crossentropy(y_true, y_pred):
    """mask掉非预测部分
    """
    y_true = K.reshape(y_true, K.shape(y_true)[:2])
    y_mask = K.cast(K.greater(y_true, 0.5), K.floatx())
    loss = K.sparse_categorical_crossentropy(y_true, y_pred)
    loss = K.sum(loss * y_mask) / K.sum(y_mask)
    return loss[None, None]


def evaluate(data):
    """线下评测函数
    """
    Y_true, Y_pred = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true)[:, 0, 5:7]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        y_true = y_true[:, 0] - 5
        Y_pred.extend(y_pred)
        Y_true.extend(y_true)
    return roc_auc_score(Y_true, Y_pred)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_score = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_score = evaluate(valid_generator)
        if val_score > self.best_val_score:
            self.best_val_score = val_score
            model.save_weights(f"{model_save_dir}/best_model.weights")
        logger.info(f'val_score: {val_score:.2f}, {self.best_val_score:.2f}')


def predict_to_file(model, data_generator, out_file):
    """预测结果到文件
    """
    F = open(out_file, 'w')
    for x_true, _ in tqdm(test_generator):
        y_pred = model.predict(x_true)[:, 0, 5:7]
        y_pred = y_pred[:, 1] / (y_pred.sum(axis=1) + 1e-8)
        for p in y_pred:
            F.write('%f\n' % p)
    F.close()


if __name__ == "__main__":
    # 加载预训练模型
    model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_mlm=True,
        keep_tokens=[0, 100, 101, 102, 103, 100, 100] + keep_tokens[:len(tokens)]
    )

    model.compile(loss=masked_crossentropy, optimizer=Adam(1e-5))
    model.summary()

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=args.epoches,
        callbacks=[evaluator]
    )

    model.load_weights(f"{model_save_dir}/best_model.weights")
    predict_fn = f"../prediction_result/result_match_mlm_{datetime.now().strftime('%m%d')}.tsv"
    predict_to_file(model, test_generator, predict_fn)
