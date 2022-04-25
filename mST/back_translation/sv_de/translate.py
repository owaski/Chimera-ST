import os
import csv

import torch as th
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

from trainer import Trainer, TrainerArgs

def load_df_from_tsv(path: str):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )

path = '/mnt/raid0/siqi/datasets/covost2/sv-SE/train_st_sv-SE_en.tsv'
sv_df = load_df_from_tsv(path)
en_ref = sv_df['tgt_text'].tolist()

device = 'cuda'

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de").to(device)

batch_size = 500
de_trans = []
for idx in tqdm(range(0, len(en_ref), batch_size)):
    input = en_ref[idx : idx + batch_size]
    tokenized_input = tokenizer(input, padding=True, return_tensors="pt").to(device)
    output = model.generate(**tokenized_input)
    translation = tokenizer.batch_decode(output, skip_special_tokens=True)
    de_trans.extend(translation)

dest_dir = '/mnt/raid0/siqi/datasets/covost2/sv-SE/16kHz_de'
os.makedirs(dest_dir, exist_ok=True)
sv_audio = sv_df['audio'].tolist()
for de, audio in tqdm(zip(de_trans, sv_audio), total=len(de_trans)):
    os.system('tts --text "{}" --vocoder_name vocoder_models/de/thorsten/wavegrad --model_name tts_models/de/thorsten/tacotron2-DCA --use_cuda 1 --out_path {}'.format(
        de, os.path.join(dest_dir, audio)
    ))