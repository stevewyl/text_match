"""
SBERT with Online Constract Loss
Refer from: https://sbert.net/examples/training/quora_duplicate_questions/README.html
"""

import logging
import math
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer, losses, evaluation
from sentence_transformers import LoggingHandler
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from utils import load_data, get_train_valid

# 日志
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# 模型
model = SentenceTransformer("distiluse-base-multilingual-cased")
model_save_path = 'output/sbert_OnlineConstrativeLoss-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(model_save_path, exist_ok=True)
num_epochs = 5
train_batch_size = 64

# 损失函数
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
margin = 0.5
train_loss = losses.OnlineContrastiveLoss(
    model=model, distance_metric=distance_metric, margin=margin)

# 数据
texts, labels = load_data("data/gaiic_track3/round1_train.tsv", has_label=True)
x_train, x_valid, y_train, y_valid = get_train_valid(texts, labels)
train_samples = [InputExample(texts=x, label=int(y)) for x, y in zip(x_train, y_train)]
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# 评估器
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
for (text_a, text_b), label in zip(x_valid, y_valid):
    dev_sentences1.append(text_a)
    dev_sentences2.append(text_b)
    dev_labels.append(int(label))
binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
    dev_sentences1, dev_sentences2, dev_labels)

# 训练
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logger.info("Warmup-steps: {}".format(warmup_steps))
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=binary_acc_evaluator,
    epochs=num_epochs,
    evaluation_steps=500,
    warmup_steps=warmup_steps,
    output_path=model_save_path
)
