import os
import argparse

from chimera.prepare_data.data_utils import load_df_from_tsv, save_df_to_tsv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phon-src', type=str, required=True)
    parser.add_argument('--tsv', type=str, required=True)
    args = parser.parse_args()
    return args

def replace(args):
    df = load_df_from_tsv(args.tsv)
    with open(args.phon_src, 'r') as r:
        phon_src_text = [line.strip() for line in r.readlines() if line.strip() != ""]
    df['src_text'] = phon_src_text
    basename = os.path.basename(args.tsv)
    dirname = os.path.dirname(args.tsv)
    comps = basename.split('.')
    comps[0] = comps[0] + '_phon'
    new_base_name = '.'.join(comps)
    save_df_to_tsv(df, os.path.join(dirname, new_base_name))

def main():
    args = parse_args()
    replace(args)

if __name__ == '__main__':
    main()