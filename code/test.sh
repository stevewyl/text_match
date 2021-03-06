MODEL=$1
MODEL_VERSION=$2
TEST_FILE=$3
NFOLDS=${4:-1}

case $MODEL in
    sbert) python test/sbert.py $MODEL_VERSION $TEST_FILE $NFOLDS ;;
    cross) python test/cross_encoder.py $MODEL_VERSION $TEST_FILE $NFOLDS ;;
    whiten) python test/bert_whitening.py $MODEL_VERSION $TEST_FILE ;;
esac