import sys
from datetime import datetime

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances


def load_data(filename):
    sent1, sent2 = [], []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        text_a, text_b = line.split("\t", 1)
        sent1.append(text_a)
        sent2.append(text_b)
    return sent1, sent2


def predict(model, data, batch_size=1024):
    sentences1, sentences2 = data
    embeddings1 = model.encode(sentences1, batch_size=batch_size,
                               show_progress_bar=True, convert_to_numpy=True)
    embeddings2 = model.encode(sentences2, batch_size=batch_size,
                               show_progress_bar=True, convert_to_numpy=True)
    cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
    return cosine_scores


# 加载模型
model_version = sys.argv[1]
model = SentenceTransformer(f"../user_data/model_data/sbert-{model_version}")

# 读取数据
test_file = sys.argv[2]
sent1, sent2 = load_data(f"../tcdata/oppo_breeno_round1_data/{test_file}")

# 模型预测
scores = predict(model, (sent1, sent2))

# 输出结果
timestamp = datetime.now().strftime("%m%d")
with open(f"../prediction_result/result_sbert_{timestamp}.tsv", "w") as fw:
    for score in scores:
        fw.write(str(score) + "\n")
