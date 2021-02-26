# text_match
Text Match Model Zoos

## TODO

1. Predict Module
2. EarlyStopping Callback
3. Cross Validation Module

## Performance

### Oppo-Xiaobu-100k

Model              | Loss/Output         | PLM                         | Best-Epoch-Step | Dev-F1 | Dev-AUC | TestA-F1
------------------ | :-----------------: | :-------------------------: | :-------------: | :----: | :-----: | :-------
Cross-Encoder      | CELoss              | Roberta-wwm-ext             | 3-1500          | 88.80  |         |
Cross-Encoder      | CELoss              | MacBERT-base                | 4-500           | 89.96  |         |
SBERT              | OnlineConstractLoss | Distiluse-base-multilingual | 4-1000          | 87.13  |         |
BERT-Whitening     | Last2-Avg           | Roberta-wwm-ext             | -               | 62.86  |         |