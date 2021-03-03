MODEL=$1
PLM=${2:-"roberta"}
BATCH_SIZE=${3:-64}

case $MODEL in
    sbert) python train/sbert.py -p $PLM -b $BATCH_SIZE ;;
    cross) python train/cross_encoder.py -p $PLM -b $BATCH_SIZE ;;
    cross_scratch) python train/cross_encoder.py -p bert_scratch -b $BATCH_SIZE --scratch ;;
    sbert_scratch) python train/sbert.py -p bert_scratch -b $BATCH_SIZE --scratch ;;
    whiten) python train/bert_whitening.py -w -p $PLM ;;
esac