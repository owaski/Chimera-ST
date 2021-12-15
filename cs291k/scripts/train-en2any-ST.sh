#!/bin/bash

# prerequisits and environment variables
export name="st_mt_cif_pretrained_noshrink"
export ST_SAVE_DIR="$SAVE_ROOT/$name"
export MT_SAVE_DIR="$SAVE_ROOT/mt"
export SAVE_DIR=$ST_SAVE_DIR
export WAVE2VEC_DIR="$SAVE_ROOT/pretrained"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $ST_SAVE_DIR $MT_SAVE_DIR $WAVE2VEC_DIR $WMT_ROOT $MUSTC_ROOT
# resume="True"
reset_optimizer="--reset-optimizer"
max_updates=150000
num_gpus=1
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
    --train-subset train-tiny_wave --valid-subset dev_wave \
    --max-tokens 2000000 --max-source-positions 2000000 \
    --save-dir $SAVE_DIR --save-interval 10 \
    --tensorboard-logdir $TB_DIR/$name \
    --config-yaml config_wave.yaml \
    \
    --criterion cs291k_criterion --label-smoothing 0.1 \
    --report-accuracy --loss-ratio 1 1 0 0 0 \
    \
    --arch cs291k_model_base --share-decoder-input-output-embed \
    --w2v2-model-path $WAVE2VEC_DIR/$pretrained_ckpt \
    \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 1e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --max-update $max_updates --warmup-updates 4000 \
    $reset_optimizer \
    \
    --update-freq $(expr 8 / $num_gpus) --num-workers 1 \
    --ddp-backend no_c10d \
    --best-checkpoint-metric st_loss \
    --fp16 --seed $seed --no-shrink
