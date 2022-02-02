import os.path as op

from nbformat import write

class DataConfigWriter(object):
    DEFAULT_VOCAB_FILENAME = "dict.txt"

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for S2T data config")
        self.yaml = yaml
        self.yaml_path = yaml_path
        self.config = {}

    def flush(self):
        with open(self.yaml_path, "w") as f:
            self.yaml.dump(self.config, f)

    def set_audio_root(self, audio_root=""):
        self.config["audio_root"] = audio_root

    def set_vocab_filename(self, vocab_filename="dict.txt"):
        self.config["vocab_filename"] = vocab_filename

    def set_bpe_tokenizer(self, bpe_tokenizer):
        self.config["bpe_tokenizer"] = bpe_tokenizer

    def set_src_vocab_filename(self, vocab_filename="dict.txt"):
        self.config["src_vocab_filename"] = vocab_filename

    def set_src_bpe_tokenizer(self, bpe_tokenizer):
        self.config["src_bpe_tokenizer"] = bpe_tokenizer

    def set_lang_list_filename(self, lang_list_filename):
        self.config["lang_list_filename"] = lang_list_filename

    def set_prepend_tgt_lang_tag(self, flag=True):
        self.config["prepend_tgt_lang_tag"] = flag

    def set_prepend_src_lang_tag(self, flag=True):
        self.config["prepend_src_lang_tag"] = flag

    def set_use_audio_input(self, flag=True):
        self.config["use_audio_input"] = flag
    
    


def gen_config_yaml(
    data_root,
    spm_filename,
    vocab_filename,
    lang_list_filename,
    yaml_filename="config.yaml",
    prepend_tgt_lang_tag=False,
    prepend_src_lang_tag=False,
    use_audio_input=False,
):
    data_root = op.abspath(data_root)
    writer = DataConfigWriter(op.join(data_root, yaml_filename))
    writer.set_audio_root(op.abspath(data_root))
    writer.set_vocab_filename(vocab_filename)
    writer.set_src_vocab_filename(vocab_filename)
    writer.set_lang_list_filename(lang_list_filename)
    bpe_tokenizer = {
        "bpe": "sentencepiece",
        "sentencepiece_model": spm_filename
    }
    writer.set_bpe_tokenizer(bpe_tokenizer)
    writer.set_src_bpe_tokenizer(bpe_tokenizer)
    if prepend_tgt_lang_tag:
        writer.set_prepend_tgt_lang_tag(True)
    if prepend_src_lang_tag:
        writer.set_prepend_src_lang_tag(True)
    if use_audio_input:
        writer.set_use_audio_input(True)
    writer.flush()

    