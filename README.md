# text_match
Text Match Model Zoos

## TODO

1. Predict Module
2. EarlyStopping Callback
3. Cross Validation Module -> 挑选模型预测错误的样本 -> 错误分析

## Usage

```bash
cd code
bash train.sh
```

## Performance

### Oppo-Xiaobu-100k

#### Supervised

Model              | Loss                   | PLM                         | Best-Epoch-Step | Dev-F1 | Dev-AUC | TestA-F1
------------------ | :-------------------:  | :-------------------------: | :-------------: | :----: | :-----: | :-------
Cross-Encoder      | CELoss                 | Roberta-wwm-ext             | 3-1500          | 88.80  |         |
Cross-Encoder      | CELoss                 | MacBERT-base                | 4-500           | 89.96  |         |
SBERT              | OnlineConstractiveLoss | Distiluse-base-multilingual | 4-1000          | 87.13  |         |

#### Unsupervised

Model              | Output              | PLM                         | Dev-F1 | Dev-AUC | TestA-F1
------------------ | :-----------------: | :-------------------------: | :----: | :-----: | :-------
BERT-Whitening     | Last2-Avg           | Roberta-wwm-ext             | 62.64  | 81.52   |
BERT               | Last2-Avg           | Roberta-wwm-ext             | 54.41  | 80.29   |
BERT-Whitening     | Last2-Avg           | MacBERT-base                | 61.53  | 78.69   |
BERT-Whitening     | All                 | Roberta-wwm-ext             | 59.95  | 79.96   |
BERT-Whitening     | Last1               | Roberta-wwm-ext             | 54.40  | 80.13   |