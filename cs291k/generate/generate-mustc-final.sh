#!/bin/bash
if [[ $target == "" ]]; then
    target=de
fi
fairseq-generate ${MUSTC_ROOT}/en-$target --gen-subset tst-COMMON_wave \
    --task speech_to_text --path $1 \
    --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
    --config-yaml config_wave.yaml --lenpen 1.0 --fp16
