import os
import argparse

import numpy as np
import torch as th
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from examples.speech_to_text.data_utils import load_df_from_tsv

def main(args):
    device = th.device('cuda:{}'.format(args.rank))

    mbart = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt").to(device)
    mbart_tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
    mbart_tokenizer.src_lang = args.lang_code

    df = load_df_from_tsv(args.tsv_path)
    sentences = df['sentence'].tolist()
    n_total = len(df)
    bulk_size = int(np.ceil(n_total / args.world_size))

    start = args.rank * bulk_size
    end = min((args.rank + 1) * bulk_size, n_total)

    batch_size = 50
    tgt_texts = [] 
    with th.no_grad():
        for idx in tqdm(range(start, end, batch_size)):
            left = idx
            right = min(idx + batch_size, end)
            encoded_src_text = mbart_tokenizer(sentences[left : right], padding=True, return_tensors="pt").to(device)
            generated_tokens = mbart.generate(**encoded_src_text)
            batch_tgt_text = mbart_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            tgt_texts.extend(batch_tgt_text)
    os.makedirs('translation/{}'.format(args.lang_code), exist_ok=True)
    th.save(tgt_texts, 'translation/{}/{}.pt'.format(args.lang_code, args.rank))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv-path', type=str)
    parser.add_argument('--lang-code', type=str)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world-size', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
