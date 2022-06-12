export EXP_ID="XSTNet"
export SAVE_DIR=/mnt/data/siqiouyang/runs/mST/$EXP_ID

export max_updates=150000
export num_gpus=4
export seed=1

export train_subset=fr_en_train,de_en_train,es_en_train,fa_en_train,it_en_train,ru_en_train,pt_en_train,zh-CN_en_train,nl_en_train,et_en_train,mn_en_train,tr_en_train,ar_en_train,sv-SE_en_train,lv_en_train,sl_en_train,ta_en_train,ja_en_train,id_en_train
export valid_subset=fr_en_dev,de_en_dev,es_en_dev,fa_en_dev,it_en_dev,ru_en_dev,pt_en_dev,zh-CN_en_dev,nl_en_dev,et_en_dev,mn_en_dev,tr_en_dev,ar_en_dev,sv-SE_en_dev,lv_en_dev,sl_en_dev,ta_en_dev,ja_en_dev,id_en_dev

fairseq-train $COVOST2_ROOT \
  --task multilingual_triplet_align_adv_task \
  --train-subset $train_subset --valid-subset $valid_subset \
  --max-tokens 400000 --max-tokens-valid 3200000 --max-source-positions 400000 \
  --save-dir $SAVE_DIR --save-interval-updates 1000 --save-interval 1 \
  --keep-last-epochs 10 --keep-interval-updates 10 \
  --tensorboard-logdir $TB_DIR/$EXP_ID \
  --config-yaml config_mST.yaml \
  \
  --criterion multilingual_triplet_align_adv_criterion --label-smoothing 0.1 \
  --report-accuracy --loss-ratio 1.0 1.0 1.0 0.0 0.0 0.0 0.0 --gamma 0.05 --use-emb --ignore-prefix-size 1 \
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
  --distributed-world-size $num_gpus \
  --update-freq $(expr 40 / $num_gpus) --num-workers 1 \
  --ddp-backend no_c10d \
  \
  --seed $seed --all-gather-list-size 131072 \
  --reset-optimizer --reset-dataloader \
  \
  --eval-bleu --eval-bleu-args '{"beam": 4, "lenpen": 1.0}' \
  --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples \
  --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $mBART50_DIR/sentence.bpe.model \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric