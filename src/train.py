from .generation import *
import tiktoken
import torch

print(__name__)
if __name__ == "__main__":
    tokenizer = tiktoken.get_encoding("gpt2")
    print("Hello World!")
    input_text = "Every effort moves you!"

    token_ids = text_to_token_ids(input_text, tokenizer)

    print(token_ids)





def calc_loss_batch(input_batch, target_batch, model, device):
  input_batch = input_batch.to(device)
  target_batch = target_batch.to(device)
  logits = model(input_batch)
  loss = torch.nn.functional.cross_entropy(logits.flatten(0,1), target_batch.flatten(), ignore_index=-100)
  return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
  total_loss = 0
  if len(data_loader) == 0:
    print("Data loader has length 0.")
    return float("nan")
  elif num_batches is None:
    num_batches = len(data_loader)
  else:
    num_batches = min(num_batches, len(data_loader))
  for i, (input_batch, target_batch) in enumerate(data_loader):
    if i >= num_batches:
      break
    loss = calc_loss_batch(input_batch, target_batch, model, device)
    total_loss += loss.item()
  return total_loss / num_batches


@torch.no_grad()
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
  model.eval()
  model.to(device)

  train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
  val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

  model.train()
  return train_loss, val_loss

@torch.no_grad()
def generate_text_simple(model, context, max_new_tokens, max_context_length):
  for _ in range(max_new_tokens):
    curr_context = context[:,-max_context_length:] # only process 1 batch of batch size of 1 usually
    logits = model(curr_context)
    logits = logits[:,-1,:]
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.argmax(probs, dim=-1, keepdim=True)  # greedy encoding
    context = torch.cat((context, next_token), dim=1)
  return context


@torch.no_grad()
def generate_and_print_sample(model, tokenizer, device, start_context):
  model.eval()
  context_size = model.pos_emb.weight.shape[0]
  encoded = text_to_token_ids(start_context, tokenizer).to(device)
  token_ids = generate_text_simple(model, encoded, 20, context_size)


  decoded_text = token_ids_to_text(token_ids, tokenizer)
  print(decoded_text)
  print("\n\nInput\n\n", start_context)
  print("\n\nModel Output\n\n", decoded_text[len(start_context):])
  model.train()


@torch.no_grad()
def generate(model, device, start_context, tokenizer, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
  model.eval()
  model.to(device)
  context_ids = text_to_token_ids(start_context, tokenizer).to(device)
  for _ in range(max_new_tokens):
    curr_context_ids = context_ids[:,-context_size:]
    logits = model(curr_context_ids)
    logits = logits[:,-1,:]
    if top_k is not None:
      top_logits, _ = torch.topk(logits, k=top_k)
      min_val = top_logits[:,-1]
      logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(device), logits)
    
    if temperature > 0.0:
      logits /= temperature
      probs = torch.softmax(logits, dim=-1)
      next_id = torch.multinomial(probs, num_samples=1)
    else:
      next_id = torch.argmax(logits, dim=-1, keepdim=True)
    if next_id == eos_id:
      break
    context_ids = torch.cat((context_ids, next_id), dim=1)
  return context_ids



def generate_text_simple(model, context, max_new_tokens, max_context_length):
  for _ in range(max_new_tokens):
    curr_context = context[:,-max_context_length:]
    with torch.no_grad():
      logits = model(curr_context)
    
    logits = logits[:,-1,:]   # obtain the last token embedding for each seq_len in each batch
    probas = torch.softmax(logits, dim=-1)
    next_token = torch.argmax(probas, dim=-1, keepdim=True)   # greedy encoding
    context = torch.cat((context, next_token), dim=1)
  
  return context


