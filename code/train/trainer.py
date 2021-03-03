import argparse
import logging
import math
import os
from datetime import datetime

from sentence_transformers import losses
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

from custom.sbert import SentenceTransformer
from custom.cross_encoder import CrossEncoder
from utils import load_data, get_train_valid, yield_train_valid
from evaluator import AUCEvaluator

arg_parser = argparse.ArgumentParser(description="Supervised Text Similarity (SBERT & Cross Encoder etc.)")
arg_parser.add_argument("-p", "--plm", type=str, default="roberta", help="pretrained language model name")
arg_parser.add_argument("-l", "--loss", type=str, default="online-contrastive", help="loss function")
arg_parser.add_argument("-e", "--epoches", type=int, default=5, help="number of training epoches")
arg_parser.add_argument("-sl", "--max_length", type=int, default=32, help="max input sequence length")
arg_parser.add_argument("-b", "--batch_size", type=int, default=64, help="training batch size")
arg_parser.add_argument("-es", "--eval_steps", type=int, default=500, help="evaluation steps")
arg_parser.add_argument("-f", "--folds", type=int, default=1, help="cross validation folds, default is 1.")
arg_parser.add_argument("-s", "--scratch", action="store_true", help="whether to train model from scratch")
arg_parser.add_argument("-i", "--interactive", action="store_true", help="use interactive or represented model")
args = arg_parser.parse_args()

# 日志
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

model_full_name = {
    "distill": "distiluse-base-multilingual-cased",
    "roberta": "hfl/chinese-roberta-wwm-ext",
    "macbert": "hfl/chinese-macbert-base",
    "bert_scratch": "./train/bert_scratch"
}
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def prepare_data(data):
    # label_map_func = lambda x: float(x) if args.loss == "cosine" and not args.interactive else lambda x: int(x)
    x_train, x_valid, y_train, y_valid = data
    # 训练数据
    train_samples = []
    for (text_a, text_b), label in zip(x_train, y_train):
        train_samples.append(InputExample(texts=[text_a, text_b], label=int(label)))
        if args.interactive:
            train_samples.append(InputExample(texts=[text_b, text_a], label=int(label)))
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)

    # 评估数据
    dev_samples = []
    for (text_a, text_b), label in zip(x_valid, y_valid):
        dev_samples.append(InputExample(texts=[text_a, text_b], label=int(label)))
    return train_dataloader, dev_samples


def get_evaluators(dev_samples, ds_name="dev"):
    evaluators = []
    if args.interactive:
        ce_evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name=ds_name)
        evaluators.append(ce_evaluator)
    else:
        binary_acc_evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, name=ds_name)
        evaluators.append(binary_acc_evaluator)
    auc_evaluator = AUCEvaluator.from_input_examples(dev_samples, interactive=args.interactive, name=ds_name)
    evaluators.append(auc_evaluator)
    seq_evaluator = SequentialEvaluator(evaluators, main_score_function=lambda scores: scores[-1])
    return seq_evaluator


def train(data):
    # 数据
    train_dataloader, dev_samples = prepare_data(data)
    # 评估器
    evaluator = get_evaluators(dev_samples)
    # 模型 & 训练
    base_module = "cross_encoder" if args.interactive else "sbert"
    if args.folds > 1:
        fold_suffix = f"-fold{i + 1}"
    model_save_path = f"../user_data/model_data/{base_module}-{args.plm}-{timestamp}{fold_suffix}"
    os.makedirs(model_save_path, exist_ok=True)
    plm_name = model_full_name[args.plm]

    warmup_steps = math.ceil(len(train_dataloader) * args.epoches * 0.1)
    logger.info("Warmup-steps: {}".format(warmup_steps))
    if args.interactive:
        model = CrossEncoder(plm_name, num_labels=1, from_scratch=args.scratch, max_length=args.max_length)
        # TODO: 默认CEloss，支持其他loss
        model.fit(
            train_dataloader=train_dataloader,
            evaluator=evaluator,
            epochs=args.epoches,
            evaluation_steps=args.eval_steps,
            warmup_steps=warmup_steps,
            output_path=model_save_path
        )
    else:
        model = SentenceTransformer(plm_name, from_scratch=args.scratch, max_length=args.max_length)
        # 损失函数
        distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE
        margin = 0.5
        if args.loss == "contrastive":
            train_loss = losses.ContrastiveLoss(model, distance_metric, margin)
        elif args.loss == "online-contrastive":
            train_loss = losses.OnlineContrastiveLoss(model, distance_metric, margin)
        elif args.loss == "cosine":
            train_loss = losses.CosineSimilarityLoss(model)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=args.epoches,
            evaluation_steps=args.eval_steps,
            warmup_steps=warmup_steps,
            output_path=model_save_path
        )


if __name__ == "__main__":
    data_fn = "train_id.tsv" if args.scratch else "train.tsv"
    texts, labels = load_data(f"../tcdata/oppo_breeno_round1_data/{data_fn}", has_label=True)

    if args.folds > 1:
        i = 0
        for data in yield_train_valid(texts, labels, args.folds):
            print(f"\nFold {i + 1}/{args.folds}")
            train(data)
    else:
        data = get_train_valid(texts, labels)
        train(data)
