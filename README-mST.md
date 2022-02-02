# To Reproduce

Prepare CoVoST 2 dataset (using zh-CN as an example)
```bash
export COVOST2_ROOT=/mnt/raid0/siqi/datasets/covost2

python mST/prepare_data/prep_covost_data.py \
  --data-root $COVOST2_ROOT --src-lang zh-CN --tgt-lang en # for single direction

bash mST/prepare_data/prepare_many_to_one.sh # prepare many to one directions
``` 

The split is in the form of {src code}_{tgt_code}_{train/dev/test}.

The config should contain language_list_filename.
prepend_tgt_lang_tag should be set to True.
prepend_src_lang_tag should be set to True.

```bash
export mBART50_DIR=/mnt/raid0/siqi/checkpoints/pretrained/mbart50.ft.n1

cp $mBART50_DIR/dict.zh_CN.txt $mBART50_DIR/dict.txt

python mST/prepare_data/gen_data_config.py --audio-root $COVOST2_ROOT \
  --spm-path $mBART50_DIR/sentence.bpe.model \
  --dict-path $mBART50_DIR/dict.txt \
  --lang-list-path $mBART50_DIR/ML50_langs.txt
```

Training:
```bash
export EXP_ID="test_20"
export SAVE_DIR=/mnt/raid0/siqi/checkpoints/$EXP_ID
export TB_DIR=tensorboard_logs
export W2V2_PATH=/mnt/raid0/siqi/checkpoints/pretrained/xlsr2_300m.pt

export max_updates=150000
export num_gpus=4
export seed=1

# exclude Catalan (ca) and Welsh (cy) since it is not in mBart50 vocab
export train_subset=fr_en_train,de_en_train,es_en_train,it_en_train,ru_en_train,zh-CN_en_train,pt_en_train,fa_en_train,et_en_train,mn_en_train,nl_en_train,tr_en_train,ar_en_train,sv-SE_en_train,lv_en_train,sl_en_train,ta_en_train,ja_en_train,id_en_train

export valid_subset=fr_en_dev,de_en_dev,es_en_dev,it_en_dev,ru_en_dev,zh-CN_en_dev,pt_en_dev,fa_en_dev,et_en_dev,mn_en_dev,nl_en_dev,tr_en_dev,ar_en_dev,sv-SE_en_dev,lv_en_dev,sl_en_dev,ta_en_dev,ja_en_dev,id_en_dev

CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-train $COVOST2_ROOT \
  --task multilingual_triplet_task \
  --train-subset $train_subset --valid-subset $valid_subset \
  --max-tokens 800000 --max-source-positions 800000 \
  --save-dir $SAVE_DIR --save-interval-updates 1000 --save-interval 1 \
  --keep-last-epochs 1 --keep-interval-updates 20 \
  --tensorboard-logdir $TB_DIR/$EXP_ID \
  --config-yaml config_mST.yaml \
  \
  --criterion multilingual_triplet_criterion --label-smoothing 0.1 \
  --report-accuracy --loss-ratio 1.0 0.2 1.0 --ignore-prefix-size 1 \
  \
  --arch xlsr_mbart50_base \
  --w2v2-model-path $W2V2_PATH \
  --mbart50-dir $mBART50_DIR \
  --cnn-subsampler \
  \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 \
  --lr 2e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0 \
  --max-update $max_updates --warmup-updates 25000 \
  \
  --update-freq $(expr 20 / $num_gpus) --num-workers 1 \
  --ddp-backend no_c10d \
  \
  --fp16 --seed $seed \
  # \
  # --eval-bleu --eval-bleu-args '{"beam": 4, "lenpen": 1.0}' \
  # --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
  # --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $mBART50_DIR/sentence.bpe.model \
  # --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  
```

Test
```bash
CUDA_VISIBLE_DEVICES=5 fairseq-generate ${COVOST2_ROOT} --gen-subset zh-CN_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/test_20/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 1000000 --max-source-positions 1000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0
```
