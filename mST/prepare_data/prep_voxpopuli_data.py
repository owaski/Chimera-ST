import os
import argparse

import torchaudio
import pandas as pd
from tqdm import tqdm

from examples.speech_to_text.data_utils import save_df_to_tsv

LANGS = [
    'bg',
    'cs',
    'da',
    'nl',
    'en',
    'et',
    'fi',
    'fr',
    'de',
    'el',
    'hu',
    'it',
    'lv',
    'lt',
    'mt',
    'pl',
    'pt',
    'ro',
    'sk',
    'sl',
    'es',
    'sv',
    'hr'
]

YEARS = {
    "10k": ["2019", "2020"]
}

MANIFEST_COLUMNS = [
    "id", "audio", "n_frames", "src_lang"
]

def main(args):
    root = os.path.join(args.data_root, "unlabelled_data")
    years = YEARS[args.subset]
    for lang in tqdm(LANGS):
        manifest = {c: [] for c in MANIFEST_COLUMNS}
        for year in years:
            dirname = os.path.join(root, lang, year)
            for audiofile in os.listdir(dirname):
                info = torchaudio.info(os.path.join(dirname, audiofile))
                manifest["id"].append(audiofile.replace('.ogg', ''))
                manifest["audio"].append('{}/{}'.format(year, audiofile))
                manifest["n_frames"].append(info.num_frames)
                manifest["src_lang"].append(lang)
        df = pd.DataFrame.from_dict(manifest)
        save_df_to_tsv(df, os.path.join(root, lang, '{}.tsv'.format(lang)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", "-d", required=True, type=str)
    parser.add_argument("--subset", required=True, type=str, choices=["10k"])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)