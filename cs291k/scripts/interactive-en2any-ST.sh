#!/bin/bash

# prerequisits and environment variables
pretrained_ckpt=wav2vec_small.pt
mkdir -p $WAVE2VEC_DIR $MUSTC_ROOT

# downloading wav2vec2 ckpt
bash chimera/tools/download_wav2vec2.sh $pretrained_ckpt $WAVE2VEC_DIR

# Train on MuST-C data
fairseq-interactive ${MUSTC_ROOT}/en-$target \
    --task cs291k_task \
    --config-yaml config_wave.yaml \
    --max-tokens 1000000 --max-source-positions 1000000 \
    --path $checkpoint
