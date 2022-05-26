import os
import pandas as pd
from tqdm import tqdm
import torchaudio

from examples.speech_to_text.data_utils import load_df_from_tsv

root = '/mnt/data/siqiouyang/datasets/cv-corpus-9.0-2022-04-27/de'
df = load_df_from_tsv('/mnt/data/siqiouyang/datasets/cv-corpus-9.0-2022-04-27/de/validated.tsv')

os.makedirs(os.path.join(root, '16kHz'), exist_ok=True)
for path in tqdm(df['path']):
    audio_path = os.path.join(root, 'clips/{}'.format(path))
    new_audio_path = os.path.join(root, '16kHz/{}.wav'.format(path[:-4]))
    if not os.path.exists(new_audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        resampled_waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        torchaudio.save(new_audio_path, resampled_waveform, sample_rate=16000)