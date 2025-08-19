import datasets
import tiktoken
from datasets import load_dataset_builder
from datasets import load_dataset
from datasets import get_dataset_split_names
from datasets import get_dataset_config_names
from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from functools import partial


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "emb_dim": 768,
    "context_length": 1024,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}


import importlib
import GPT2Base
import generation
importlib.reload(GPT2Base)
importlib.reload(generation)
from generation import *
from GPT2Base import *
model = GPTModel(GPT_CONFIG_124M)


iterable_dataset = load_dataset("gsm8k", 'socratic', streaming=True)
datasetdict = load_dataset("gsm8k", 'socratic')
tokenizer = tiktoken.get_encoding("gpt2")


torch.manual_seed(123)
dataset_train = datasetdict['train']
dataset_val = datasetdict['train']
dataset_converted = InstructionDataset(dataset_train, tokenizer)


train_ratio = 0.8
train_size = int(train_ratio * len(dataset_converted))
val_size = len(dataset_converted) - train_size
dataset_train_converted = dataset_converted[:train_size]
dataset_val_converted = dataset_converted[train_size:]



my_collate_fn = partial(custom_collate_fn, allowed_max_length=256)
train_loader = DataLoader(dataset_train_converted, batch_size=4, collate_fn=my_collate_fn, drop_last=True, shuffle=True)
val_loader = DataLoader(dataset_val_converted, batch_size=4, collate_fn=my_collate_fn, drop_last=True, shuffle=True)