# To Reproduce

Prepare CoVoST 2 dataset (using zh-CN as an example)
```bash
export COVOST2_ROOT=/mnt/raid5/siqi/datasets/covost2
export COVOST2_ROOT=/local/home/siqiouyang/work/dataset/covost2

python mST/prepare_data/prep_covost_data.py \
  --data-root $COVOST2_ROOT --src-lang zh-CN --tgt-lang en # for single direction

bash mST/prepare_data/prepare_many_to_one.sh # prepare many to one directions
``` 

The split is in the form of {src code}_{tgt_code}_{train/dev/test}.

The config should contain language_list_filename.
prepend_tgt_lang_tag should be set to True.
prepend_src_lang_tag should be set to True.


Additional data from VoxPopuli.
```bash
export VOXP_ROOT=/mnt/raid5/siqi/datasets/voxpopuli
export VOXP_ROOT=/local/home/siqiouyang/work/dataset/voxpopuli

cd /mnt/nvme/siqi/work/libraries/voxpopuli

python -m voxpopuli.download_audios --root $VOXP_ROOT --subset 10k 
python -m voxpopuli.get_unlabelled_data --root $VOXP_ROOT --subset 10k
```


```bash
export mBART50_DIR=/mnt/raid5/siqi/checkpoints/pretrained/mbart50.ft.n1
export mBART50_DIR=/local/home/siqiouyang/work/checkpoint/pretrained/mbart50.ft.n1

cp $mBART50_DIR/dict.zh_CN.txt $mBART50_DIR/dict.txt

python mST/prepare_data/gen_data_config.py --audio-root $COVOST2_ROOT \
  --spm-path $mBART50_DIR/sentence.bpe.model \
  --dict-path $mBART50_DIR/dict.txt \
  --lang-list-path $mBART50_DIR/ML50_langs.txt \
  --voxpopuli-root $VOXP_ROOT \
  --unlabeled-sampling-ratio 0.1
```

Training:
```bash
export EXP_ID="xlsr_phone_mbart_tr"
export SAVE_DIR=/mnt/raid5/siqi/checkpoints/$EXP_ID
export SAVE_DIR=/local/home/siqiouyang/work/checkpoint/$EXP_ID
export TB_DIR=tensorboard_logs
export W2V2_PATH=/mnt/raid5/siqi/checkpoints/pretrained/xlsr2_300m.pt
export W2V2_PATH=/local/home/siqiouyang/work/checkpoint/pretrained/xlsr2_300m.pt

export max_updates=150000
export num_gpus=4
export seed=1

# exclude Catalan (ca) and Welsh (cy) since it is not in mBart50 vocab
export train_subset=fr_en_train,de_en_train,es_en_train,it_en_train,ru_en_train,zh-CN_en_train,pt_en_train,fa_en_train,et_en_train,mn_en_train,nl_en_train,tr_en_train,ar_en_train,sv-SE_en_train,lv_en_train,sl_en_train,ta_en_train,ja_en_train,id_en_train # ,en_de_train,en_tr_train,en_fa_train,en_sv-SE_train,en_mn_train,en_zh-CN_train,en_sl_train,en_et_train,en_id_train,en_ar_train,en_ta_train,en_lv_train,en_ja_train

export valid_subset=fr_en_dev,de_en_dev,es_en_dev,it_en_dev,ru_en_dev,zh-CN_en_dev,pt_en_dev,fa_en_dev,et_en_dev,mn_en_dev,nl_en_dev,tr_en_dev,ar_en_dev,sv-SE_en_dev,lv_en_dev,sl_en_dev,ta_en_dev,ja_en_dev,id_en_dev # ,en_de_dev,en_tr_dev,en_fa_dev,en_sv-SE_dev,en_mn_dev,en_zh-CN_dev,en_sl_dev,en_et_dev,en_id_dev,en_ar_dev,en_ta_dev,en_lv_dev,en_ja_dev

# only EU languages
export train_subset=fr_en_train,de_en_train,es_en_train,it_en_train,pt_en_train,et_en_train,nl_en_train,sv-SE_en_train,lv_en_train,sl_en_train,fr,de,es,it,pt,et,nl,sv,lv,sl
export valid_subset=fr_en_dev,de_en_dev,es_en_dev,it_en_dev,pt_en_dev,et_en_dev,nl_en_dev,sv-SE_en_dev,lv_en_dev,sl_en_dev

# only de and sv-SE
export train_subset=sv-SE_en_train,de_en_train
export valid_subset=sv-SE_en_dev,de_en_dev


CUDA_LAUNCH_BLOCKING=1 TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_SHOW_CPP_STACKTRACES=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-train $COVOST2_ROOT \
  --task multilingual_triplet_task \
  --train-subset $train_subset --valid-subset $valid_subset \
  --max-tokens 800000 --max-source-positions 800000 \
  --save-dir $SAVE_DIR --save-interval-updates 1000 --save-interval 1 \
  --keep-last-epochs 1 --keep-interval-updates 1 \
  --tensorboard-logdir $TB_DIR/$EXP_ID \
  --config-yaml config_mST.yaml \
  \
  --criterion multilingual_triplet_criterion --label-smoothing 0.1 \
  --report-accuracy --loss-ratio 1.0 0.1 1.0 0.0 --ignore-prefix-size 1 \
  \
  --arch xlsr_mbart50_base \
  --w2v2-model-path $W2V2_PATH \
  --mbart50-dir $mBART50_DIR \
  --cnn-subsampler \
  \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 \
  --lr 2e-4 --lr-scheduler inverse_sqrt --weight-decay 0.0 --dropout 0.1 \
  --max-update $max_updates --warmup-updates 5000 \
  \
  --update-freq $(expr 20 / $num_gpus) --num-workers 1 \
  --ddp-backend no_c10d \
  \
  --fp16 --seed $seed --all-gather-list-size 32768 
  # --encoder-layer-to-remove-residual 4 8
  # \
  # --eval-bleu --eval-bleu-args '{"beam": 4, "lenpen": 1.0}' \
  # --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
  # --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $mBART50_DIR/sentence.bpe.model \
  # --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  


# for contrastive 

CUDA_VISIBLE_DEVICES=0 \
python mST/prepare_data/gen_positive_pairs.py \
  --data_dir $COVOST2_ROOT \
  --ref_langs de \
  --tgt_langs fr \
  -k 1 \
  --threshold 0.7

export train_subset=sv-SE_en_train
export valid_subset=sv-SE_en_dev

mkdir $SAVE_DIR
cp /mnt/raid5/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt $SAVE_DIR/checkpoint_last.pt

cp /mnt/raid5/siqi/checkpoints/sv_ft_5/checkpoint_best.pt $SAVE_DIR/checkpoint_last.pt

export num_gpus=1
CUDA_VISIBLE_DEVICES=0 \
fairseq-train $COVOST2_ROOT \
  --task multilingual_triplet_contrastive_task \
  --train-subset $train_subset --valid-subset $valid_subset \
  --max-tokens 800000 --max-source-positions 800000 \
  --save-dir $SAVE_DIR --save-interval-updates 5000 --save-interval 1 \
  --keep-last-epochs 1 --keep-interval-updates 1 \
  --tensorboard-logdir $TB_DIR/$EXP_ID \
  --config-yaml config_mST.yaml \
  \
  --criterion multilingual_triplet_contrastive_criterion --label-smoothing 0.1 \
  --report-accuracy --loss-ratio 0.0001 0.0 0.0 0.0001 --ignore-prefix-size 1 \
  --contrast-layer 0 --gamma 0.05 \
  \
  --arch xlsr_mbart50_base \
  --w2v2-model-path $W2V2_PATH \
  --mbart50-dir $mBART50_DIR \
  --cnn-subsampler \
  \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 \
  --lr 5e-5 --lr-scheduler inverse_sqrt --weight-decay 0.0 \
  --max-update $max_updates --warmup-updates 5000 \
  \
  --update-freq $(expr 20 / $num_gpus) --num-workers 1 \
  --ddp-backend no_c10d \
  \
  --fp16 --seed $seed --all-gather-list-size 40000 \
  --reset-optimizer --reset-dataloader


# for phone
export train_subset=de_en_train_phone,tr_en_train_phone # ,zh-CN_en_train_phone
export valid_subset=de_en_dev_phone,tr_en_dev_phone # ,zh-CN_en_dev_phone

export train_subset=tr_en_train_phone # ,zh-CN_en_train_phone
export valid_subset=tr_en_dev_phone # ,zh-CN_en_dev_phone
  
export num_gpus=4
CUDA_VISIBLE_DEVICES=0,1,2,3 \
fairseq-train $COVOST2_ROOT \
  --task multilingual_triplet_phone_task \
  --train-subset $train_subset --valid-subset $valid_subset \
  --max-tokens 400000 --max-source-positions 400000 \
  --save-dir $SAVE_DIR --save-interval-updates 5000 --save-interval 1 \
  --keep-last-epochs 1 --keep-interval-updates 1 \
  --tensorboard-logdir $TB_DIR/$EXP_ID \
  --config-yaml config_mST.yaml \
  \
  --criterion multilingual_triplet_phone_criterion --label-smoothing 0.1 \
  --report-accuracy --loss-ratio 1.0 0.1 1.0 0.0 0.0 --ignore-prefix-size 1 \
  \
  --arch xlsr_phone_mbart50_base \
  --w2v2-model-path $W2V2_PATH \
  --mbart50-dir $mBART50_DIR \
  --cnn-subsampler \
  \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 10.0 \
  --lr 5e-5 --lr-scheduler inverse_sqrt --weight-decay 0.0 \
  --max-update $max_updates --warmup-updates 5000 \
  \
  --update-freq $(expr 40 / $num_gpus) --num-workers 1 \
  --ddp-backend no_c10d \
  \
  --fp16 --seed $seed --all-gather-list-size 131072 \
  --reset-optimizer --reset-dataloader

```

Test
```bash
CUDA_VISIBLE_DEVICES=1 fairseq-generate ${COVOST2_ROOT} --gen-subset tr_en_test \
  --task multilingual_speech_to_text_phone --path /mnt/raid5/siqi/checkpoints/xlsr_phone_mbart_de_tr/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 800000 --max-source-positions 800000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0
```



export VER=std

export SRC_LANG=fr;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=de;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=es;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=fa;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=it;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=ru;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=pt;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=zh-CN;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=tr;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=ar;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=et;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=mn;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=nl;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=sv-SE;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=lv;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=sl;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=ta;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=ja;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \
export SRC_LANG=id;  fairseq-generate ${COVOST2_ROOT} --gen-subset ${SRC_LANG}_en_test \
  --task multilingual_speech_to_text --path /mnt/raid0/siqi/checkpoints/xlsr_mbart_n1/checkpoint_best.pt \
  --prefix-size 1 --max-tokens 2000000 --max-source-positions 2000000 --beam 4 --scoring sacrebleu \
  --config-yaml config_mST.yaml --lenpen 1.0 --max-len-a 0 --max-len-b 1; \