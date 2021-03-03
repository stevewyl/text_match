"""
SBERT with Online Constract Loss
Refer from: https://sbert.net/examples/training/quora_duplicate_questions/README.html
"""

import argparse
import logging
import math
import os
from datetime import datetime

from sentence_transformers import losses, evaluation
from sentence_transformers import LoggingHandler
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from custom.sbert import SentenceTransformer
from evaluator import AUCEvaluator
from utils import load_data, get_train_valid

# 日志
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

arg_parser = argparse.ArgumentParser(description="Supervised Text Similarity (SBERT)")
arg_parser.add_argument("-p", "--plm", type=str, default="distill", help="pretrained language model name")
arg_parser.add_argument("-l", "--loss", type=str, default="online-contrastive", help="loss function")
arg_parser.add_argument("-e", "--epoches", type=int, default=5, help="number of training epoches")
arg_parser.add_argument("-s", "--max_length", type=int, default=32, help="max input sequence length")
arg_parser.add_argument("-b", "--batch_size", type=int, default=64, help="training batch size")
arg_parser.add_argument("--scratch", action="store_true", help="whether to train model from scratch")
args = arg_parser.parse_args()

# 模型
model_full_name = {
    "distill": "distiluse-base-multilingual-cased",
    "roberta": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base",
    "bert_scratch": "./train/bert_scratch"
}
model = SentenceTransformer(
    model_full_name[args.plm], max_length=args.max_length, from_scratch=args.scratch)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_save_path = f"../user_data/model_data/sbert-{args.plm}-{args.loss}-{timestamp}"
os.makedirs(model_save_path, exist_ok=True)

# 损失函数
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
margin = 0.5
label_type_map = lambda x: int(x)
if args.loss == "contrastive":
    train_loss = losses.ContrastiveLoss(model, distance_metric, margin)
elif args.loss == "online-contrastive":
    train_loss = losses.OnlineContrastiveLoss(model, distance_metric, margin)
elif args.loss == "cosine":
    train_loss = losses.CosineSimilarityLoss(model)
    label_type_map = lambda x: float(x)

# 数据
if args.scratch:
    fn = "train_id.tsv"
else:
    fn = "train.tsv"
texts, labels = load_data(f"../tcdata/oppo_breeno_round1_data/{fn}", has_label=True)
x_train, x_valid, y_train, y_valid = get_train_valid(texts, labels)
train_samples = [InputExample(texts=x, label=label_type_map(y)) for x, y in zip(x_train, y_train)]
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

# 评估器
dev_sentences1 = []
dev_sentences2 = []
dev_labels = []
for (text_a, text_b), label in zip(x_valid, y_valid):
    dev_sentences1.append(text_a)
    dev_sentences2.append(text_b)
    dev_labels.append(label_type_map(label))
evaluators = []
binary_acc_evaluator = evaluation.BinaryClassificationEvaluator(
    dev_sentences1, dev_sentences2, dev_labels)
evaluators.append(binary_acc_evaluator)
auc_evaluator = AUCEvaluator(x_valid, dev_labels, False)
evaluators.append(auc_evaluator)
seq_evaluator = evaluation.SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])

# 训练
warmup_steps = math.ceil(len(train_dataloader) * args.epoches * 0.1)
logger.info("Warmup-steps: {}".format(warmup_steps))
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=seq_evaluator,
    epochs=args.epoches,
    evaluation_steps=500,
    warmup_steps=warmup_steps,
    output_path=model_save_path
)
