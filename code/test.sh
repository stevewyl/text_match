MODEL=$1
MODEL_VERSION=$2

case $MODEL in
    sbert) python test/sbert.py $MODEL_VERSION;;
    cross) python test/cross_encoder.py $MODEL_VERSION ;;
    whiten) python test/bert_whitening.py $MODEL_VERSION ;;
esac