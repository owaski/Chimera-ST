import argparse

from chimera.prepare_data.data_utils import gen_vocab

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)    
    parser.add_argument('--vocab-size', type=int, required=True)    
    parser.add_argument('--model-prefix', type=str, required=True)    
    args = parser.parse_args()
    return args

def main(args):
    gen_vocab(args.input, args.model_prefix, 'unigram', args.vocab_size)

if __name__ == '__main__':
    args = parse_args()
    main(args)