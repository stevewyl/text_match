"""
Cross Encoder with CE Loss
Refer from: https://sbert.net/examples/training/cross-encoder/README.html
"""

import argparse
import logging
import math
import os
import sys
from datetime import datetime

from sentence_transformers import LoggingHandler, evaluation
# from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from Cross_Encoder import CrossEncoder
from evaluator import AUCEvaluator
from utils import load_data, get_train_valid

# 日志
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser(description="Supervised Text Similarity (Cross-Encoder)")
arg_parser.add_argument("-p", "--plm", type=str, default="distill", help="pretrained language model name")
arg_parser.add_argument("-e", "--epoches", type=int, default=5, help="number of training epoches")
arg_parser.add_argument("-b", "--batch_size", type=int, default=64, help="training batch size")
arg_parser.add_argument("-s", "--max_length", type=int, default=40, help="max input sequence length")
arg_parser.add_argument("--scratch", action="store_true", help="whether to train model from scratch")
args = arg_parser.parse_args()

# 模型
model_full_name = {
    "roberta": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base",
    "bert_scratch": "./train/bert_scratch"
}
model = CrossEncoder(model_full_name[args.plm], num_labels=1, max_length=args.max_length, from_scratch=args.scratch)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = f"../user_data/model_data/cross_encoder-{args.plm}-{timestamp}"
os.makedirs(model_save_path, exist_ok=True)

# 数据
if args.scratch:
    fn = "train_id.tsv"
else:
    fn = "train.tsv"
texts, labels = load_data(f"../tcdata/oppo_breeno_round1_data/{fn}", has_label=True)
x_train, x_valid, y_train, y_valid = get_train_valid(texts, labels)
train_samples = []
for (text_a, text_b), label in zip(x_train, y_train):
    train_samples.append(InputExample(texts=[text_a, text_b], label=int(label)))
    train_samples.append(InputExample(texts=[text_b, text_a], label=int(label)))
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

# 评估器
dev_samples = [InputExample(texts=x, label=int(y)) for x, y in zip(x_valid, y_valid)]
evaluators = []
ce_evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name="dev")
evaluators.append(ce_evaluator)
auc_evaluator = AUCEvaluator.from_input_examples(dev_samples, name="dev")
evaluators.append(auc_evaluator)
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

# 训练
warmup_steps = math.ceil(len(train_dataloader) * args.epoches * 0.1)
logger.info("Warmup-steps: {}".format(warmup_steps))
model.fit(
    train_dataloader=train_dataloader,
    evaluator=seq_evaluator,
    epochs=args.epoches,
    evaluation_steps=500,
    warmup_steps=warmup_steps,
    output_path=model_save_path
)
