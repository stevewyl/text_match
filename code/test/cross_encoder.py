import sys
from datetime import datetime

import numpy as np
from sentence_transformers import CrossEncoder


class Predictor:
    def __init__(self, model_version, reverse=True):
        model_path = f"../user_data/model_data/cross_encoder-{model_version}"
        self.model = CrossEncoder(model_path, num_labels=1, max_length=32)
        self.reverse = reverse

    def load_data(self, filename):
        sentence_pairs = []
        for line in open(f"../tcdata/oppo_breeno_round1_data/{filename}", "r", encoding="utf-8"):
            line = line.strip()
            text_a, text_b = line.split("\t", 1)
            sentence_pairs.append([text_a, text_b])
            if self.reverse:
                sentence_pairs.append([text_b, text_a])
        return sentence_pairs

    def predict(self, data, batch_size=256):
        pred_scores = self.model.predict(data, batch_size=batch_size,
                                         convert_to_numpy=True, show_progress_bar=True)
        if self.reverse:
            pred_scores = np.array([pred_scores[i:i+2] for i in range(0, pred_scores.shape[0], 2)])
            pred_scores = np.mean(pred_scores, axis=1)
        timestamp = datetime.now().strftime("%m%d")
        with open(f"../prediction_result/result_cross_encoder_{timestamp}.tsv", "w") as fw:
            for score in pred_scores:
                fw.write(str(score) + "\n")


if __name__ == "__main__":
    model_version = sys.argv[1]
    test_file = sys.argv[2]
    nfolds = int(sys.argv[3])

    if nfolds < 2:
        predictor = Predictor(model_version)
        data = predictor.load_data(test_file)
        predictor.predict(data)
    else:
        for i in range(1, nfolds + 1):
            model_version = model_version + f"-fold{i}"
            