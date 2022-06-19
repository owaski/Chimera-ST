export EXP_ID="baseline_0"
export SAVE_DIR=/mnt/data/siqiouyang/runs/mST/$EXP_ID

export max_updates=150000
export num_gpus=1
export seed=1

export train_subset=de-0_en_train
export valid_subset=de_en_dev

export CUDA_VISIBLE_DEVICES=5
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

python mST/scripts/average_checkpoints.py \
  --inputs \
  /mnt/data/siqiouyang/runs/mST/$EXP_ID/checkpoint_6_16000.pt \
  /mnt/data/siqiouyang/runs/mST/$EXP_ID/checkpoint_6_17000.pt \
  /mnt/data/siqiouyang/runs/mST/$EXP_ID/checkpoint_7_18000.pt \
  /mnt/data/siqiouyang/runs/mST/$EXP_ID/checkpoint_7_19000.pt \
  /mnt/data/siqiouyang/runs/mST/$EXP_ID/checkpoint_7_20000.pt \
  --output /mnt/data/siqiouyang/runs/mST/$EXP_ID/checkpoint_avg.pt

for lang in fr de es fa it ru pt zh-CN nl et mn tr ar sv-SE lv sl ta ja id
do
  echo Processing $lang
  CUDA_VISIBLE_DEVICES=0 fairseq-generate ${COVOST2_ROOT} --gen-subset ${lang}_en_test \
    --task multilingual_speech_to_text --path /mnt/data/siqiouyang/runs/mST/$EXP_ID/checkpoint_avg.pt \
    --prefix-size 1 --max-tokens 3200000 --max-source-positions 400000 --beam 4 --scoring sacrebleu \
    --config-yaml config_mST.yaml --lenpen 1.0 --results-path \
    /home/siqiouyang/work/projects/mST/mST/translations/$EXP_ID/$lang
done
