import sys

from sentence_transformers import CrossEncoder


def load_data(filename):
    sentence_pairs = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        text_a, text_b = line.split("\t", 1)
        sentence_pairs.append([text_a, text_b])
    return sentence_pairs


def predict(model, data, batch_size=1024):
    pred_scores = model.predict(data, batch_size=batch_size,
                                convert_to_numpy=True, show_progress_bar=True)
    return pred_scores


# 加载模型
model_version = sys.argv[1]
model = CrossEncoder(f"../user_data/model_data/cross_encoder-{model_version}",
                     num_labels=1, max_length=40)

# 读取数据
sentence_pairs = load_data("../tcdata/oppo_breeno_round1_data/testA.tsv")

# 模型预测
scores = predict(model, sentence_pairs)

# 输出结果
with open("../prediction_result/result_cross_encoder.tsv", "w") as fw:
    for score in scores:
        fw.write(str(score) + "\n")
