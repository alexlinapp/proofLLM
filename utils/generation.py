import torch
def text_to_token_ids(input, tokenizer):
  encoded = tokenizer.encode(input, allowed_special={'<|endoftext|>'})
  encoded_tensor = torch.tensor(encoded).unsqueeze(0)   # adds batch dimension
  return encoded_tensor

def token_ids_to_text(input, tokenizer):
  decoded = input.squeeze(0)
  return tokenizer.decode(decoded.tolist())









def generate_text_simple(model, context, max_new_tokens, max_context_length):
  for _ in range(max_new_tokens):
    curr_context = context[:,-max_context_length:]
    with torch.no_grad():
      logits = model(curr_context)
    
    logits = logits[:,-1,:]   # obtain the last token embedding for each seq_len in each batch
    probas = torch.softmax(logits, dim=-1)
    next_token = torch.argmax(probas, dim=-1, keepdim=True)
    context = torch.cat((context, next_token), dim=1)
  
  return context