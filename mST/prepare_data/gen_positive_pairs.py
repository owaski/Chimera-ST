import os
import argparse

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

from examples.speech_to_text.data_utils import load_df_from_tsv

def main(args):
    device = 'cuda'

    covost2_dir = args.data_dir
    ref_df = None
    for ref_lang in args.ref_langs:
        df = load_df_from_tsv(os.path.join(covost2_dir, ref_lang, 'train_st_{}_en.tsv'.format(ref_lang)))
        if ref_df is None:
            ref_df = df
        else:
            ref_df = pd.concat([ref_df, df])

    model = SentenceTransformer('sentence-transformers/LaBSE').to(device)
    batch_size = 2000
    ref_texts = ref_df['tgt_text'].tolist()
    with th.inference_mode():
        ref_features = []
        for i in tqdm(range(0, len(ref_texts), batch_size), 'en'):
            # inputs = tokenizer(text, return_tensors="pt").to('cuda')
            # outputs = model(**inputs)
            # features.append(outputs.last_hidden_state.cpu())
            outputs = model.encode(ref_texts[i:i+batch_size])
            ref_features.append(outputs)
        ref_features = np.concatenate(ref_features, axis=0)
        ref_features = ref_features / np.linalg.norm(ref_features, axis=-1).reshape(-1, 1)

    index = faiss.IndexFlatIP(ref_features.shape[-1])
    index.add(ref_features)

    ref_audios = ref_df['audio'].tolist()
    ref_langs = ref_df['src_lang'].tolist()
    for tgt_lang in args.tgt_langs:
        tgt_df = load_df_from_tsv(os.path.join(covost2_dir, tgt_lang, 'train_st_{}_en.tsv'.format(tgt_lang)))
        batch_size = 2000
        tgt_texts = tgt_df['tgt_text'].tolist()
        with th.inference_mode():
            tgt_features = []
            for i in tqdm(range(0, len(tgt_texts), batch_size), desc=tgt_lang):
                outputs = model.encode(tgt_texts[i:i+batch_size])
                tgt_features.append(outputs)
            tgt_features = np.concatenate(tgt_features, axis=0)
            tgt_features = tgt_features / np.linalg.norm(tgt_features, axis=-1).reshape(-1, 1)
        
        sims, ids = index.search(tgt_features, k=args.k)
        positive_refs = []
        for i in range(len(tgt_texts)):
            refs = []
            for j in range(args.k):
                if sims[i, j] > args.threshold:
                    refs.append(os.path.join(covost2_dir, ref_langs[ids[i, j]], '16kHz', ref_audios[ids[i, j]]))
                    # print(tgt_texts[i], ref_texts[ids[i, j]], sims[i, j], ref_df['src_lang'][ids[i, j]], ids[i, j], end='\n', sep=' | ')
            positive_refs.append(refs)
        th.save(positive_refs, os.path.join(covost2_dir, tgt_lang, 'train_refs.pt'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--ref_langs', type=str, nargs='+')
    parser.add_argument('--tgt_langs', type=str, nargs='+')
    parser.add_argument('-k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.5)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)