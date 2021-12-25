import os
import argparse

from chimera.prepare_data.data_utils import load_df_from_tsv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv', type=str, required=True)
    args = parser.parse_args()
    return args

def extract_src(args):
    df = load_df_from_tsv(args.tsv)
    dirname = os.path.dirname(args.tsv)
    with open(os.path.join(dirname, 'train_src.txt'), 'w') as w:
        w.write('\n'.join(df['src_text'].tolist()))

def main():
    args = parse_args()
    extract_src(args)

if __name__ == '__main__':
    main()