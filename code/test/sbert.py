import sys
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances


class Predictor:
    def __init__(self, model_version):
        model_path = f"../user_data/model_data/sbert-{model_version}"
        self.model = SentenceTransformer(model_path)

    def load_data(self, filename):
        sent1, sent2 = [], []
        for line in open(f"../tcdata/oppo_breeno_round1_data/{filename}", "r", encoding="utf-8"):
            line = line.strip()
            text_a, text_b = line.split("\t", 1)
            sent1.append(text_a)
            sent2.append(text_b)
        return sent1, sent2

    def predict(self, filename, batch_size=1024):
        sentences1, sentences2 = self.load_data(filename)
        embeddings1 = self.model.encode(sentences1, batch_size=batch_size,
                                        show_progress_bar=True, convert_to_numpy=True)
        embeddings2 = self.model.encode(sentences2, batch_size=batch_size,
                                        show_progress_bar=True, convert_to_numpy=True)
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        return cosine_scores

    def save(self, pred_scores):
        timestamp = datetime.now().strftime("%m%d")
        with open(f"../prediction_result/result_sbert_{timestamp}.tsv", "w") as fw:
            for score in pred_scores:
                fw.write(str(score) + "\n")


if __name__ == "__main__":
    base_model_version = sys.argv[1]
    test_file = sys.argv[2]
    if len(sys.argv) > 3:
        nfolds = int(sys.argv[3])
    else:
        nfolds = 1

    if nfolds < 2:
        predictor = Predictor(base_model_version)
        scores = predictor.predict(test_file)
    else:
        scores = []
        for i in range(1, nfolds + 1):
            model_version = base_model_version + f"-fold{i}"
            predictor = Predictor(model_version)
            scores.append(predictor.predict(test_file))
        scores = np.mean(np.array(scores).T, axis=1)
    predictor.save(scores)
