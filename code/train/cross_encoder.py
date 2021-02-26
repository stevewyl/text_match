"""
Cross Encoder with CE Loss
Refer from: https://sbert.net/examples/training/cross-encoder/README.html
"""

import logging
import math
import os
import sys
from datetime import datetime

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers import LoggingHandler
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from utils import load_data, get_train_valid

plm = sys.argv[1]
plm_names = {
    "roberta": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base"
}

# 日志
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# 模型
model = CrossEncoder(plm_names[plm], num_labels=1, max_length=40)
model_save_path = '../output/cross_encoder_roberta_CELoss-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(model_save_path, exist_ok=True)
num_epochs = 5
train_batch_size = 64

# 数据
texts, labels = load_data("../../train.tsv", has_label=True)
x_train, x_valid, y_train, y_valid = get_train_valid(texts, labels)
train_samples = []
for (text_a, text_b), label in zip(x_train, y_train):
    train_samples.append(InputExample(texts=[text_a, text_b], label=int(label)))
    train_samples.append(InputExample(texts=[text_b, text_a], label=int(label)))
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# 评估器
dev_samples = [InputExample(texts=x, label=int(y)) for x, y in zip(x_valid, y_valid)]
evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name="dev")

# 训练
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)
logger.info("Warmup-steps: {}".format(warmup_steps))
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    evaluation_steps=500,
    warmup_steps=warmup_steps,
    output_path=model_save_path
)
