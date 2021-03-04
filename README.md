# text_match

Text Match Model Zoos

## TODO

1. Cross Validation Module
2. SBERT-Scratch
3. Wrong Analysis
4. Multi-task SBERT
5. Keyword-BERT

## Usage

```bash
cd code
# Training model
# supported model names
# sbert, cross_encoder, bert_scratch, whiten
# supported PLM
# roberta, macbert, distill, bert_scratch
bash train.sh {MODEL_NAME} {PLM} {BATCH_SIZE}

# Predictions
bash test.sh {MODEL} {MODEL_VERSION} {TEST_FILE}
```

## Performance

### Oppo-Xiaobu-100k

#### Supervised

Model              | Loss                   | PLM                         | Best-Epoch-Step | Dev-F1 | Dev-AUC | TestA-F1
------------------ | :-------------------:  | :-------------------------: | :-------------: | :----: | :-----: | :-------
Cross-Encoder      | CELoss                 | Roberta-wwm-ext             | 3-2000          | 89.36  | 97.43   |
Cross-Encoder      | CELoss                 | MacBERT-base                | 3-2500          | 89.86  | 97.69   |
Cross-Encoder      | CELoss                 | BERT-scratch                | 6-2500          | 76.18  | 87.86   |
SBERT              | CosineSimilarityLoss   | Distiluse-base-multilingual | 4-2500          | 83.61  | 94.39   |
SBERT              | ConstractiveLoss       | Distiluse-base-multilingual | 4-1500          | 86.05  | 95.93   |
SBERT              | OnlineConstractiveLoss | Distiluse-base-multilingual | 4-500           | 86.34  | 95.97   |
SBERT              | OnlineConstractiveLoss | MacBERT-base                | 3-1500          | 88.32  | 96.99   |
SBERT              | OnlineConstractiveLoss | Roberta-wwm-ext             | 4-2500          | 88.26  | 96.93   |
SBERT              | OnlineConstractiveLoss | BERT-scratch                | 8-2500          | 77.86  | 90.73   |

#### Unsupervised

Model              | Output              | PLM                         | Dev-F1 | Dev-AUC | TestA-F1
------------------ | :-----------------: | :-------------------------: | :----: | :-----: | :-------
BERT-Whitening     | Last2-Avg           | Roberta-wwm-ext             | 62.64  | 75.14   |
BERT               | Last2-Avg           | Roberta-wwm-ext             | 54.41  | 78.02   |
BERT-Whitening     | Last2-Avg           | MacBERT-base                | 61.53  | 74.79   |
BERT-Whitening     | All                 | Roberta-wwm-ext             | 64.03  | 76.41   |
BERT-Whitening     | Last1               | Roberta-wwm-ext             | 54.40  | 80.13   |
