import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
  def __init__(self, d_in, d_out, context_length, num_heads, dropout=0.1, qkv_bias=False):
    super().__init__()
    assert d_out % num_heads == 0, "d_out must be divsible by num_heads"
    self.head_dim = d_out // num_heads
    self.num_heads = num_heads
    self.W_q = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_k = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.W_v = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
    self.out_proj = torch.nn.Linear(d_out, d_out)
    self.dropout = nn.Dropout(dropout)
    self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

  def forward(self, x):
    b, num_tokens, d_in = x.shape
    q = self.W_q(x).view(b, num_tokens, self.num_heads, self.head_dim)
    k = self.W_k(x).view(b, num_tokens, self.num_heads, self.head_dim)
    v = self.W_v(x).view(b, num_tokens, self.num_heads, self.head_dim)

    q = q.transpose(1,2)    # (b, num_tokens, self.num_heads, self.head_dim) -> (b, self.num_heads, num_tokens, self.head_dim)
    k = k.transpose(1,2)
    v = v.transpose(1,2)

    atten_score = q @ k.transpose(2,3) # to end up with a (b, self.num_heads, num_tokens, num_tokens) atten_score
    atten_score.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
    atten_weights = torch.softmax(atten_score / (k.shape[-1] ** 0.5), dim=-1)
    atten_weights = self.dropout(atten_weights)
    context_vec = (atten_weights @ v).transpose(1,2)    # transposing a tensor of shape (b, self.num_heads, num_tokens, self.head_dim)

    # contiguous mean when we access row by row (last dim by last dim) elements we get should be contiguous in memory. Hence transpose ruins this
    context_vec = context_vec.contiguous().view(b, num_tokens, self.head_dim * self.num_heads)  # view can only be used on vector in contiguous memory. Transpose makes it non-contiguous hence use .contiguous

    context_vec = self.out_proj(context_vec)
    return context_vec

class LayerNorm(nn.Module):
  def __init__(self, d_in, eps=1e-5):
    super().__init__()
    self.eps = eps
    self.scale = torch.nn.Parameter(torch.ones(d_in))
    self.bias = torch.nn.Parameter(torch.zeros(d_in))
  def forward(self, x):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True)
    norm_x = (x - mean) / torch.sqrt(var + self.eps)
    return norm_x * self.scale + self.bias

class GELU(nn.Module):
  def __init__(self):
    super().__init__()
  def forward(self, x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
  def forward(self, x):
    return self.layers(x)


class TransformerBlock(nn.Module):
  def __init__(self, cfg):
    super().__init__()
    self.norm1 = LayerNorm(cfg["emb_dim"])
    self.norm2 = LayerNorm(cfg["emb_dim"])
    self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"], cfg["n_heads"])
    self.ff = FeedForward(cfg)
    self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
  
  def forward(self, x):
    shortcut = x
    x = self.norm1(x)
    x = self.att(x)
    x = self.drop_shortcut(x)
    x = x + shortcut
    shortcut = x
    x = self.norm2(x)
    x = self.ff(x)
    x = self.drop_shortcut(x)
    x = x + shortcut
    return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = torch.nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = torch.nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)


    def forward(self, x):
        bs, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits