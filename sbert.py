from sentence_transformers import SentenceTransformer
from sentence_transformers import InputExample, losses, evaluation
from torch.utils.data import DataLoader

from utils import load_data, get_train_valid

# 数据
texts, labels = load_data("data/gaiic_train.tsv")
x_train, x_valid, y_train, y_valid = get_train_valid(texts, labels)
train_examples = [InputExample(texts=x, label=y) for x, y in zip(x_train, y_train)]
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# 模型
model = SentenceTransformer("distiluse-base-multilingual-cased")
train_loss = losses.CosineSimilarityLoss(model)

# 训练
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

losses.