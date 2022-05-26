import os
import re
import random
from argparse import Namespace, ArgumentParser
from click import Argument

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb

from tqdm import tqdm
import torchaudio
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import textgrids
import sentencepiece as spm

from fairseq.data import Dictionary
from fairseq.models.mST.w2v2_phone_transformer import W2V2Transformer
from fairseq.data.audio.multilingual_triplet_v2_phone_dataset import (
    MultilingualTripletDataConfig,
    MultilingualTripletDataset,
    MultilingualTripletDatasetCreator
)
from fairseq.data.audio.speech_to_text_dataset import get_features_or_waveform, _collate_frames
from examples.speech_to_text.data_utils import load_df_from_tsv
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE, SentencepieceConfig
from examples.speech_recognition.data.data_utils import padding_mask_to_lengths


parser = ArgumentParser()
parser.add_argument('--ckpt', type=str)
parser.add_argument('--name', type=str)
parser.add_argument('--lr', type=float)
parser.add_argument('--device', type=str)
cmd_args = parser.parse_args()

device = cmd_args.device

zh_root = '/mnt/data/siqiouyang/datasets/covost2/zh-CN'

args = Namespace()
task = Namespace()

def load_dict(vocab_filename):
    _dict_path = vocab_filename
    if not os.path.isfile(_dict_path):
        raise FileNotFoundError(f"Dict not found: {_dict_path}")
    _dict = Dictionary.load(_dict_path)
    for code in codes:
        _dict.add_symbol(MultilingualTripletDataset.LANG_TAG_TEMPLATE.format(code))
    _dict.add_symbol('<mask>')
    return _dict

lang_list_filename = '/mnt/data/siqiouyang/runs/mST/pretrained/mbart50.ft.n1/ML50_langs.txt'
vocab_filename = '/mnt/data/siqiouyang/runs/mST/pretrained/mbart50.ft.n1/dict.txt'
phone_vocab_filename = '/mnt/data/siqiouyang/datasets/covost2/phone_dict.txt'


codes = MultilingualTripletDataset.get_lang_codes(lang_list_filename)
dict = load_dict(vocab_filename)
with open(phone_vocab_filename, 'r') as r:
    phone_list = [l.strip() for l in r.readlines() if l.strip() != '']
    phone_dict = {l: idx + 1 for idx, l in enumerate(phone_list)} # leave 0 as blank
    phone_list = ['|'] + phone_list

task.src_dict = task.tgt_dict = dict
task.phone_dict = phone_dict

args.w2v2_model_path = '/mnt/data/siqiouyang/runs/mST/pretrained/xlsr2_300m.pt'
args.mbart50_dir = '/mnt/data/siqiouyang/runs/mST/pretrained/mbart50.ft.n1'

model = W2V2Transformer.build_model(args, task)

ckpt_path = cmd_args.ckpt
ckpt = load_checkpoint_to_cpu(ckpt_path)

model.load_state_dict(ckpt["model"], strict=False)
model = model.to(device)
model.eval()

sp = spm.SentencePieceProcessor()
sp.Load('/mnt/data/siqiouyang/runs/mST/pretrained/mbart50.ft.n1/sentence.bpe.model')

def match(pieces, sp_ids, sp_pieces):
    j = 0
    ids = []
    for piece in pieces:
        while piece not in sp_pieces[j]:
            j += 1
            if j == len(sp_pieces):
                return -1
        ids.append(sp_ids[j])
        j += 1
    return ids

class ASRDataset(Dataset):
    def __init__(self, root, split):
        df = load_df_from_tsv(os.path.join(root, '{}_st_zh-CN_en.tsv'.format(split)))

        self.audio_paths = []
        self.segmentations = []
        self.tokenss = []

        for id, transcript in zip(tqdm(df['id']), df['src_text']):
            if os.path.exists(os.path.join(root, '16kHz/align_sp', '{}.TextGrid'.format(id))):
                audio_path = os.path.join(root, '16kHz', '{}.wav'.format(id))

                grids = textgrids.TextGrid('{}/16kHz/align_sp/{}.TextGrid'.format(zh_root, id))
                duration = torchaudio.info(audio_path).num_frames / 16000
                intervals = th.tensor([(grid.xmin, grid.xmax) for grid in grids['words'] if grid.text != '']) / duration
                
                pieces = [grid.text for grid in grids['words'] if grid.text != '']
                tokens = match(pieces, sp.Encode(transcript), sp.EncodeAsPieces(transcript))

                if tokens != -1:
                    self.segmentations.append(intervals)
                    self.audio_paths.append(audio_path)
                    self.tokenss.append(tokens)

                    assert len(tokens) == intervals.size(0)
                else:
                    print(transcript, pieces)
    
    def __getitem__(self, idx):
        return self.audio_paths[idx], self.segmentations[idx], self.tokenss[idx]

    def __len__(self):
        return len(self.audio_paths)


train_dataset = ASRDataset(zh_root, 'train')
dev_dataset = ASRDataset(zh_root, 'dev')

token_mask = th.zeros(model.encoder.text_embedding.weight.size(0), dtype=th.bool)
for dataset in [train_dataset, dev_dataset]:
    for tokens in dataset.tokenss:
        for tok in tokens:
            token_mask[tok] = True
token_mask[-1] = True

def asr_collate_fn(samples):
    audio_paths = [ap for ap, _, _ in samples]
    sources = [
        th.from_numpy(get_features_or_waveform(
            ap,
            need_waveform=True,
            sample_rate=16000,
        )).float() for ap in audio_paths
    ]
    n_frames = th.tensor([source.size(0) for source in sources], dtype=th.long).to(device)
    frames = _collate_frames(sources, True).to(device)

    segmentations = [seg.to(device) for _, seg, _  in samples]

    tokens = [th.tensor(t, dtype=th.long).to(device) for _, _, t in samples]
    ntokens = sum([t.size(0) for t in tokens])
    
    out = {
        "net_input": {
            "src_tokens": frames,
            "src_lengths": n_frames,
        },
        "segmentations": segmentations,
        "tokens": tokens,
        "ntokens": ntokens
    }
    return out

train_dataloader = DataLoader(train_dataset, batch_size=10, collate_fn=asr_collate_fn)
dev_dataloader = DataLoader(dev_dataset, batch_size=10, collate_fn=asr_collate_fn)

def compute_asr_loss(speech_encoder_out, inputs):
    speech_embs = speech_encoder_out['x']

    speech_embs = mlp(speech_embs)

    logits = th.matmul(speech_embs, model.encoder.text_embedding.weight.T.detach())

    logits[:, :, ~token_mask] = -1e4

    bsz = logits.size(0)

    padding_mask = speech_encoder_out['padding_mask']
    lens = padding_mask_to_lengths(padding_mask)

    segmentations = inputs['segmentations']

    cat_logits = []
    cat_labels = []

    blank_label = logits.size(-1) - 1

    batch_tokens = inputs['tokens']

    for idx in range(bsz):
        length = lens[idx]
        cat_logits.append(logits[idx, :length])

        labels = th.zeros(length, dtype=th.long).to(device) + blank_label

        tokens = batch_tokens[idx]

        seg = segmentations[idx]
        scaled_seg = (seg * length).long()
        for token, (l, r) in zip(tokens, scaled_seg):
            labels[l : r] = token
        
        cat_labels.append(labels)

    flat_logits = th.cat(cat_logits, dim=0)
    flat_labels = th.cat(cat_labels, dim=0)

    loss = F.cross_entropy(flat_logits, flat_labels, reduction='sum')

    ntokens = flat_logits.size(0)

    return loss, ntokens

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layer):
        super(MLP, self).__init__()

        self.layers = []
        for idx in range(n_layer):
            self.layers.append(nn.Linear(n_hidden if idx > 0 else n_input, n_hidden))
        self.layers = nn.ModuleList(self.layers)

        self.final_proj = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        x = self.final_proj(x)
        return x

mlp = MLP(1024, 2048, 1024, 2).to(device)
optimizer = th.optim.Adam(mlp.parameters(), lr=cmd_args.lr)

wandb.init(project="ST", entity="owaski", name=cmd_args.name)

loss_fn = compute_asr_loss

def eval(dataloader):
    mlp.eval()
    iterator = tqdm(dataloader)
    sum_loss = 0.
    sum_ntokens = 0
    with th.no_grad():
        for inputs in iterator:
            speech_encoder_out = model.encoder.forward_speech(**inputs["net_input"])
            loss, ntokens = loss_fn(speech_encoder_out, inputs)                
            sum_loss += loss
            sum_ntokens += ntokens
    print('eval loss {:.2f}'.format(sum_loss / sum_ntokens))
    wandb.log({'eval_loss': sum_loss / sum_ntokens})

def run_epoch(dataloader):
    mlp.train()
    iterator = tqdm(dataloader)

    sum_loss = 0.
    sum_ntokens = 0
    for inputs in iterator:
        with th.no_grad():
            speech_encoder_out = model.encoder.forward_speech(**inputs["net_input"])

        optimizer.zero_grad()
        
        loss, ntokens = loss_fn(speech_encoder_out, inputs)

        wandb.log({'train_loss': loss / ntokens})
            
        sum_loss += loss.item()
        sum_ntokens += ntokens

        loss = loss / ntokens

        loss.backward()
        optimizer.step()
        
        iterator.set_description('train_loss: {:.2f}'.format(loss.item()))
    print('train loss {:.2f}'.format(sum_loss / sum_ntokens))

try:
    n_epoch = 100
    for _ in range(n_epoch):
        run_epoch(train_dataloader)
        eval(dev_dataloader)
except Exception as e:
    print(e)
    th.save(mlp.state_dict(), 'ckpts/{}.pt'.format(cmd_args.name))
