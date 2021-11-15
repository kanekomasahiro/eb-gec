seed=$1

cpu_num=`grep -c ^processor /proc/cpuinfo`
echo $cpu_num

FAIRSEQ_DIR=../knnmt/fairseq_cli
DATA_DIR=../data
PROCESSED_DIR=../process/$seed
MODEL_DIR=../model/$seed
EVAL_DIR=../../eval_gec
beam=5


if [ ! -e $PROCESSED_DIR/bin_test ]; then
  echo Creating preprocess test data
  subword-nmt apply-bpe -c $PROCESSED_DIR/trg.bpe < $DATA_DIR/test.src > $PROCESSED_DIR/test.src
    cp $PROCESSED_DIR/test.src $PROCESSED_DIR/test.trg
  
  python -u $FAIRSEQ_DIR/preprocess.py \
    --source-lang src \
    --target-lang trg \
    --trainpref $PROCESSED_DIR/train \
    --testpref $PROCESSED_DIR/test \
    --srcdict $PROCESSED_DIR/bin/dict.src.txt \
    --tgtdict $PROCESSED_DIR/bin/dict.trg.txt \
    --destdir $PROCESSED_DIR/bin_test \
    --workers 20 \
    --tokenizer space 
fi

remove_subword="--remove-bpe"
master_subeord="--subword subword_nmt --subwordcodes ${PROCESSED_DIR}/trg.bpe"

OUTPUT_DIR=../output/$seed
if [ ! -e $PROCESSED_DIR/dstores_test ]; then
  mkdir -p $PROCESSED_DIR/dstores_test
  echo Creating test data dstores
  python ../knnmt/master_script.py \
    --log-folder ../log/index.txt \
    --slurm-name knn \
    --bytes-per-token 2 \
    --model $MODEL_DIR/checkpoint_best.pt \
    $master_subeord \
    --save-data $PROCESSED_DIR/bin_test \
    --save-example $PROCESSED_DIR/dstores_test/example \
    --binfile train.src-trg.trg.bin \
    --num-shards 1 \
    --dstore-mmap $PROCESSED_DIR/dstores_test/index_only \
    --num-for-training 1000000 \
    --code-size 64 \
    --ncentroids 4096 \
    --train-index $PROCESSED_DIR/dstores_test/index_only.4096.index.trained \
    --save-job \
    --merge-dstore-job \
    --val $PROCESSED_DIR/dstores_test/val \
    --train-index-job  

  python ../knnmt/master_script.py \
    --log-folder logs/index.txt \
    --slurm-name knn \
    --bytes-per-token 2 \
    --model $MODEL_DIR/checkpoint_best.pt \
    $master_subeord \
    --save-data $PROCESSED_DIR/bin_test \
    --save-example $PROCESSED_DIR/dstores_test/example \
    --binfile train.src-trg.trg.bin \
    --num-shards 1 \
    --dstore-mmap $PROCESSED_DIR/dstores_test/index_only \
    --num-for-training 1000000 \
    --code-size 64 \
    --ncentroids 4096 \
    --train-index $PROCESSED_DIR/dstores_test/index_only.4096.index.trained \
    --faiss-index $PROCESSED_DIR/dstores_test/index_only.4096.index \
    --write-merged-index $PROCESSED_DIR/dstores_test/index_only.4096.index \
    --corpus-identifiers knn \
    --add-keys-job \
    --val $PROCESSED_DIR/dstores_test/val \
    --merge-index-job
fi

echo Generating test data with knn
mkdir -p $OUTPUT_DIR
python $FAIRSEQ_DIR/generate.py $PROCESSED_DIR/bin_test \
  --path $MODEL_DIR/checkpoint_best.pt \
  --batch-size 32 \
  --beam ${beam} \
  --nbest 1 \
  $remove_subword \
  --source-lang src \
  --target-lang trg \
  --knnmt \
  --k 16 \
  --num-workers 20 \
  --indexfile $PROCESSED_DIR/dstores_test/index_only.4096.index \
  --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
  --knn-keytype last_ffn_input \
  --knn-embed-dim 1024 \
  --no-load-keys \
  --knn-temp 1000 \
  --knn-sim-func do_not_recomp_l2 \
  --lmbda 0.5 \
  --example $PROCESSED_DIR/dstores_test/example \
  --val $PROCESSED_DIR/dstores_test/val \
  --use-faiss-only \
  --dstore-filename $PROCESSED_DIR/dstores_test/index_only.subset \
  > $OUTPUT_DIR/test.nbest.tok

cat $OUTPUT_DIR/test.nbest.tok | grep "^H" | python -c "import sys, numpy; x = sys.stdin.readlines(); x = ' '.join([ x[i] for i in range(len(x)) if (i % 1 == 0) ]); x = x.split('\n')[:-1]; idx = numpy.argsort([int(l.strip().split('\t')[0][2:]) for l in x]).tolist(); x = '\n'.join([x[i].split('\t')[2] for i in idx]); print(x)" | cut -f3 > $OUTPUT_DIR/test.best.tok

