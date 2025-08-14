import tiktoken
import importlib
import torch
import transformers
from huggingface_hub import notebook_login
import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# print("tiktoken version:", importlib.metadata.version('tiktoken'))
# notebook_login()

# dataset = load_dataset("entfane/professor-mathematics", split="train")
# train_data = dataset.select(range(20))

# train_data = InstructionDataset(train_data, tokenizer)

# torch.manual_seed(123)
# train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=custom_collate_fn)
# inputs, targets = next(iter(train_loader))
# inputs[0][-7:], targets[0][-7:]

# print(type(dataset))
# print(isinstance(dataset, datasets.dataset_dict.DatasetDict))
# print(dataset[0]['question'], "\n\n")
# encoded_ids = tokenizer.encode(dataset[0]['answer'])

# tokenizer = tiktoken.get_encoding("gpt2")

# input = ("<|endoftext|>")
# output = tokenizer.encode(input, allowed_special={"<|endoftext|>"})
# strings = tokenizer.decode(output)
# print(strings)



'''
Dataset stored internally as python list
'''



class InstructionDataset(Dataset):
  def __init__(self, dataset, tokenizer, max_length=1024):
    self.input_ids = []
    if (isinstance(dataset, datasets.arrow_dataset.Dataset)):
      for entry in dataset:
        formatted_entry = format_input(entry)
        input_id = tokenizer.encode(formatted_entry, allowed_special={"<|endoftext|>"})
        if (len(input_id) > max_length):
          continue
        self.input_ids.append(input_id)
    else:
      print("Not datasets.arrow_dataset.Dataset class. Did not add")
  
  def __len__(self):
    return len(self.input_ids)
  def __getitem__(self, idx):
    return self.input_ids[idx]

def format_input(input) -> str:
  return ("###Question:\n" + input['question'] + "\n\n###Answer:\n" + input['answer']) 



def custom_collate_fn(batch, pad_token_id=50256,
                      ignore_index=-100,
                      allowed_max_length=None,
                      device="cpu"):
  batch_max_length = max(len(item) + 1 for item in batch)
  if allowed_max_length is not None:
    batch_max_length = min(batch_max_length, allowed_max_length+1)


  inputs_lst, targets_lst = [], []
  for item in batch:
    new_item = item.copy()
    new_item += [pad_token_id]
    padded = new_item + ([pad_token_id] * (batch_max_length - len(new_item)))
    inputs = torch.tensor(padded[:-1]);
    targets = torch.tensor(padded[1:])


    mask = targets == pad_token_id
    indices = torch.nonzero(mask).squeeze()
    if indices.numel() > 1:
      targets[indices[1:]] = ignore_index
    if allowed_max_length is not None:
      inputs = inputs[:allowed_max_length]
      targets = targets[:allowed_max_length]


    inputs_lst.append(inputs)
    targets_lst.append(targets)
  inputs_tensor = torch.stack(inputs_lst, dim=0).to(device)
  targets_tensor = torch.stack(targets_lst, dim=0).to(device)
  return inputs_tensor, targets_tensor