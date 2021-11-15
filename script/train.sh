seed=$1

FAIRSEQ_DIR=../knnmt/fairseq_cli
DATA_DIR=../data
PROCESSED_DIR=../process/$seed
MODEL_DIR=../model/$seed
num_operations=8000
beam=5
cpu_num=`grep -c ^processor /proc/cpuinfo`


mkdir -p $PROCESSED_DIR

# Preprocessing

if [ -e $PROCESSED_DIR/bin ]; then
  echo Processed file already exists
else
  echo Creating processed file
  mkdir -p $PROCESSED_DIR/bin
  subword-nmt learn-bpe -s $num_operations < $DATA_DIR/train.trg > $PROCESSED_DIR/trg.bpe
  subword-nmt apply-bpe -c $PROCESSED_DIR/trg.bpe < $DATA_DIR/train.src > $PROCESSED_DIR/train.src
  subword-nmt apply-bpe -c $PROCESSED_DIR/trg.bpe < $DATA_DIR/train.trg > $PROCESSED_DIR/train.trg
  subword-nmt apply-bpe -c $PROCESSED_DIR/trg.bpe < $DATA_DIR/dev.src > $PROCESSED_DIR/dev.src
  subword-nmt apply-bpe -c $PROCESSED_DIR/trg.bpe < $DATA_DIR/dev.trg > $PROCESSED_DIR/dev.trg

  python -u $FAIRSEQ_DIR/preprocess.py \
    --source-lang src \
    --target-lang trg \
    --trainpref $PROCESSED_DIR/train \
    --validpref $PROCESSED_DIR/dev \
    --testpref $PROCESSED_DIR/dev \
    --destdir $PROCESSED_DIR/bin \
    --workers $cpu_num \
    --tokenizer space 
fi

# Training

mkdir -p $MODEL_DIR

python -u $FAIRSEQ_DIR/train.py $PROCESSED_DIR/bin \
  --save-dir $MODEL_DIR \
  --source-lang src \
  --target-lang trg \
  --log-format simple \
  --max-epoch 20 \
  --arch transformer_vaswani_wmt_en_de_big \
  --max-tokens 4096 \
  --optimizer adam \
  --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 \
  --lr-scheduler inverse_sqrt \
  --warmup-updates 4000 \
  --warmup-init-lr 1e-07 \
  --min-lr 1e-09 \
  --dropout 0.3 \
  --clip-norm 0.0 \
  --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy \
  --label-smoothing 0.1 \
  --num-workers $cpu_num \
  --keep-best-checkpoints 5 \
  --no-epoch-checkpoints \
  --seed $seed

