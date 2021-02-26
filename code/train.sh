MODELS=("sbert" "cross" "whiten")
for model_name in ${MODELS[@]}; do
    case $model_name in
    sbert) python train/sbert.py ;;
    cross) python train/cross_encoder.py roberta ;;
    whiten) python train/bert_whitening.py roberta home ;;
    esac
done