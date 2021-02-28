# text_match
Text Match Model Zoos

## TODO

1. Keyword-BERT
2. Multi-task SBERT
3. EarlyStopping Callback
4. Cross Validation Module -> 挑选模型预测错误的样本 -> 错误分析

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
Cross-Encoder      | CELoss                 | Roberta-wwm-ext             | 2-1500          | 89.71  |         |
Cross-Encoder      | CELoss                 | MacBERT-base                | 4-500           | 89.96  |         |
SBERT              | ConstractiveLoss       | Distiluse-base-multilingual | 4-500           | 86.09  |         |
SBERT              | OnlineConstractiveLoss | Distiluse-base-multilingual | 4-1000          | 86.38  |         |
SBERT              | OnlineConstractiveLoss | Roberta-wwm-ext             | 4-2500          | 88.58  |         |

#### Unsupervised

Model              | Output              | PLM                         | Dev-F1 | Dev-AUC | TestA-F1
------------------ | :-----------------: | :-------------------------: | :----: | :-----: | :-------
BERT-Whitening     | Last2-Avg           | Roberta-wwm-ext             | 62.64  | 75.14   |
BERT               | Last2-Avg           | Roberta-wwm-ext             | 54.41  | 80.29   |
BERT-Whitening     | Last2-Avg           | MacBERT-base                | 61.53  | 74.79   |
BERT-Whitening     | All                 | Roberta-wwm-ext             | 59.95  | 79.96   |
BERT-Whitening     | Last1               | Roberta-wwm-ext             | 54.40  | 80.13   |
