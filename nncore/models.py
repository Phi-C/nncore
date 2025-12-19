import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.
    """

    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head**-0.5

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, seq, d_model
        qkv = self.qkv(x)  # (B,T,3C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B,T,C)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)  # (B,nh,T,dh)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # causal mask
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B,nh,T,T)
        att = att.masked_fill(self._causal_mask(T, x.device), float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)

        out = att @ v  # (B,nh,T,dh)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )


class MLP(nn.Module):
    """
    Standard Silu MLP Block.
    """

    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(mlp_ratio * d_model * 2 / 3)  # 常见近似
        self.fc1 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.fc2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(x1) * x2
        return self.dropout(self.fc2(x))


class Block(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, mlp_ratio: int = 4, dropout: float = 0.0
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(n_heads, d_model, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    """
    Tiny transformer for experiments.
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 8,
        max_seq_len: int = 512,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, mlp_ratio, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重初始化
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"seq len {T} exceeds max {self.max_seq_len}"
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ):
        """
        Autoregressive code completion.
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_crop = (
                idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]
            )
            logits, _ = self(idx_crop)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx


# ------------------- Demo -------------------
if __name__ == "__main__":
    model = TinyTransformer(vocab_size=10000)
    x = torch.randint(0, 10000, (2, 64))  # 批量 2，长度 64
    logits, loss = model(x, x)  # 自回归训练，target=输入右移一位即可
    print("logits shape:", logits.shape, "loss:", loss.item())

    # 采样
    prompt = torch.zeros((1, 1), dtype=torch.long)
    sample = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=40)
    print("sample idx:", sample)
