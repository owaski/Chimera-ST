#!/bin/bash

# prerequisits and environment variables
export name="debug"
export ST_SAVE_DIR="$SAVE_ROOT/$name"
export MT_SAVE_DIR="$SAVE_ROOT/mt"
export SAVE_DIR=$ST_SAVE_DIR
export WAVE2VEC_DIR="$SAVE_ROOT/pretrained"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $ST_SAVE_DIR $MT_SAVE_DIR $WAVE2VEC_DIR $WMT_ROOT $MUSTC_ROOT
resume="True"
reset_optimizer="--reset-optimizer"
max_updates=150000
num_gpus=8
target=de
seed=1

# downloading wav2vec2 ckpt
bash chimera/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# loading MT pre-trained ckpt
if [[ $resume == "True" ]]; then
    reset_optimizer=""
else
    cp $MT_SAVE_DIR/checkpoint_best.pt $ST_SAVE_DIR/checkpoint_last.pt
fi

# Train on MuST-C data
fairseq-train ${MUSTC_ROOT}/en-$target \
    --task cs291k_task \
    --train-subset train_wave --valid-subset debug   \
    --max-tokens 2000000 --max-source-positions 2000000 \
    --save-dir $SAVE_DIR --save-interval-updates 1 --save-interval 1 \
    --keep-last-epochs 1 --keep-interval-updates 20 \
    --tensorboard-logdir $TB_DIR/$name \
    --config-yaml config_wave.yaml \
    \
    --criterion cs291k_criterion --label-smoothing 0.1 \
    --report-accuracy --loss-ratio 1 0.2 1 1 0 \
    \
    --arch cs291k_model_base --share-decoder-input-output-embed \
    --w2v2-model-path $WAVE2VEC_DIR/$pretrained_ckpt \
    \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 \
    --lr 0. --lr-scheduler inverse_sqrt --weight-decay 0.0000 \
    --max-update $max_updates --warmup-updates 25000 \
    $reset_optimizer \
    \
    --update-freq $(expr 8 / $num_gpus) --num-workers 1 \
    --ddp-backend no_c10d \
    \
    --fp16 --seed $seed \
    --align-after-encoder 0
    # --cnn-subsampler
    # --eval-bleu --eval-bleu-args '{"beam": 4, "lenpen": 1.0}' \
    # --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
    # --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path ${spm_model} \
    # --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    
