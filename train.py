import torch
import torch.nn as nn

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def get_or_build_tokenizer():
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
    except FileNotFoundError:
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        trainer = WordLevelTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        dataset = load_dataset("wmt16", "ro-en", split="train")
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train_from_iterator(dataset["translation"], trainer=trainer)
        tokenizer.save(tokenizer_path)
    return tokenizer