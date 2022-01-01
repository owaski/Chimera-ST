#!/bin/bash

# prerequisits and environment variables
export MT_SAVE_DIR="$SAVE_ROOT/mt_phon"
export SAVE_DIR=$MT_SAVE_DIR
export WAVE2VEC_DIR="$SAVE_ROOT/pretrained"
pretrained_ckpt=wav2vec_small.pt
mkdir -p $MT_SAVE_DIR $WAVE2VEC_DIR $MUSTC_ROOT $WMT_ROOT

num_gpus=8
seed=1
dataset=wmt14
max_updates=500000
target=de

# downloading wav2vec2 ckpt
bash cs291k/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# WMT-MUSTC joint data and spm
TEXT=$WMT_ROOT/phon
spm_model=$TEXT/spm_unigram10000_wave.model
spm_dict=$TEXT/spm_unigram10000_wave.txt
# fairseq-preprocess \
#     --source-lang en --target-lang $target \
#     --trainpref $TEXT/train --validpref $TEXT/valid \
#     --testpref $TEXT/test,$TEXT/mustc-tst-COMMON \
#     --destdir $WMT_ROOT/bin --thresholdtgt 0 --thresholdsrc 0 \
#     --srcdict $spm_dict --tgtdict $spm_dict \
#     --workers 100

# Train on WMT data
fairseq-train $TEXT/bin \
    --task translation \
    --train-subset train --valid-subset valid \
    --save-dir $SAVE_DIR \
    --save-interval 1 \
    --keep-last-epochs 1 \
    --tensorboard-logdir $TB_DIR/mt_phon \
    --max-tokens 4096 \
    \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $spm_model \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    \
    --arch cs291k_model_base --share-decoder-input-output-embed \
    --w2v2-model-path $WAVE2VEC_DIR/$pretrained_ckpt \
    --dropout 0.1 \
    \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
    --max-update $max_updates --warmup-updates 4000 \
    --fp16 \
    \
    --update-freq $(expr 8 / $num_gpus) --num-workers 1 \
    --ddp-backend no_c10d \
    --seed $seed