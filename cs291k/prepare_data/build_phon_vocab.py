import os
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phon-src', type=str, required=True)
    args = parser.parse_args()
    return args

def build_vocab(args):
    with open(args.phon_src, 'r') as r:
        phon_src_text = [line.strip() for line in r.readlines() if line.strip() != ""]
    vocab = defaultdict(int)
    for line in phon_src_text:
        for token in line.split(' '):
            vocab[token] += 1
    dirname = os.path.dirname(args.phon_src)
    with open(os.path.join(dirname, 'src_phon.txt'), 'w') as w:
        for token, freq in vocab.items():
            w.write('{} {}\n'.format(token, freq))

def main():
    args = parse_args()
    build_vocab(args)

if __name__ == '__main__':
    main()