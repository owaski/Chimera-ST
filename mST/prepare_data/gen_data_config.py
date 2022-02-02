import argparse

from data_utils import gen_config_yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-root', type=str, required=True)
    parser.add_argument('--spm-path', type=str, required=True)
    parser.add_argument('--dict-path', type=str, required=True)
    parser.add_argument('--lang-list-path', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    gen_config_yaml(
        args.audio_root,
        args.spm_path,
        args.dict_path,
        args.lang_list_path,
        yaml_filename='config_mST.yaml',
        prepend_tgt_lang_tag=True,
        prepend_src_lang_tag=True,    
        use_audio_input=True
    )
    
if __name__ == '__main__':
    main()