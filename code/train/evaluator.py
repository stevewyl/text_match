import logging
from typing import List
import numpy as np
import os
import csv

from sentence_transformers.readers import InputExample
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import paired_cosine_distances

logger = logging.getLogger(__name__)


class AUCEvaluator:
    """
    This evaluator can be used with the CrossEncoder/SentenceTransformer class.
    Given sentence pairs and binary labels (0 and 1),
    it compute the the best possible auc score
    """
    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], interactive: bool, name: str=''):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.interactive = interactive
        self.name = name

        self.csv_file = "AUCEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "auc"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("AUCEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        if self.interactive:
            pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)
        else:
            sentences1 = [pair[0] for pair in self.sentence_pairs]
            sentences2 = [pair[1] for pair in self.sentence_pairs]
            embeddings1 = model.encode(sentences1, batch_size=1024,
                                    show_progress_bar=False, convert_to_numpy=True)
            embeddings2 = model.encode(sentences2, batch_size=1024,
                                    show_progress_bar=False, convert_to_numpy=True)

            pred_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)

        auc = roc_auc_score(self.labels, pred_scores) * 100

        logger.info(f"AUC:           {auc:.4f}")

        if output_path is not None:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, auc])

        return auc
