"""PyTorch GPT-2-style transformer with explicit causal attention and NumPy weight I/O."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bench.spec import LN_EPS, ModelSpec


class CausalSelfAttention(nn.Module):
    """Multi-head causal attention written explicitly with einsum and FP32 softmax."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.n_head = spec.n_head
        self.head_dim = spec.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(spec.d_model, 3 * spec.d_model)
        self.proj = nn.Linear(spec.d_model, spec.d_model)
        mask = torch.tril(torch.ones(spec.seq_len, spec.seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, spec.seq_len, spec.seq_len), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(d_model, dim=-1)
        q = q.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k).float() * self.scale
        scores = scores.masked_fill(~self.causal_mask[:, :, :seq_len, :seq_len], -1e30)
        attn = F.softmax(scores, dim=-1).to(v.dtype)
        y = torch.einsum("bhqk,bhkd->bhqd", attn, v)
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_model)
        return self.proj(y)


class MLP(nn.Module):
    """Two-layer feed-forward block using tanh-approximated GELU."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.fc1 = nn.Linear(spec.d_model, spec.mlp_dim)
        self.fc2 = nn.Linear(spec.mlp_dim, spec.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class TransformerBlock(nn.Module):
    """Pre-LN transformer block."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(spec.d_model, eps=LN_EPS)
        self.attn = CausalSelfAttention(spec)
        self.ln2 = nn.LayerNorm(spec.d_model, eps=LN_EPS)
        self.mlp = MLP(spec)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def load_numpy(self, params: dict[str, np.ndarray], prefix: str = "") -> None:
        with torch.no_grad():
            self.ln1.weight.copy_(torch.from_numpy(params[f"{prefix}ln1.w"]))
            self.ln1.bias.copy_(torch.from_numpy(params[f"{prefix}ln1.b"]))
            self.attn.qkv.weight.copy_(torch.from_numpy(params[f"{prefix}attn.qkv.w"].T))
            self.attn.qkv.bias.copy_(torch.from_numpy(params[f"{prefix}attn.qkv.b"]))
            self.attn.proj.weight.copy_(torch.from_numpy(params[f"{prefix}attn.proj.w"].T))
            self.attn.proj.bias.copy_(torch.from_numpy(params[f"{prefix}attn.proj.b"]))
            self.ln2.weight.copy_(torch.from_numpy(params[f"{prefix}ln2.w"]))
            self.ln2.bias.copy_(torch.from_numpy(params[f"{prefix}ln2.b"]))
            self.mlp.fc1.weight.copy_(torch.from_numpy(params[f"{prefix}mlp.fc1.w"].T))
            self.mlp.fc1.bias.copy_(torch.from_numpy(params[f"{prefix}mlp.fc1.b"]))
            self.mlp.fc2.weight.copy_(torch.from_numpy(params[f"{prefix}mlp.fc2.w"].T))
            self.mlp.fc2.bias.copy_(torch.from_numpy(params[f"{prefix}mlp.fc2.b"]))

    def to_numpy(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {
            f"{prefix}ln1.w": self.ln1.weight.detach().cpu().numpy().copy(),
            f"{prefix}ln1.b": self.ln1.bias.detach().cpu().numpy().copy(),
            f"{prefix}attn.qkv.w": self.attn.qkv.weight.detach().cpu().numpy().T.copy(),
            f"{prefix}attn.qkv.b": self.attn.qkv.bias.detach().cpu().numpy().copy(),
            f"{prefix}attn.proj.w": self.attn.proj.weight.detach().cpu().numpy().T.copy(),
            f"{prefix}attn.proj.b": self.attn.proj.bias.detach().cpu().numpy().copy(),
            f"{prefix}ln2.w": self.ln2.weight.detach().cpu().numpy().copy(),
            f"{prefix}ln2.b": self.ln2.bias.detach().cpu().numpy().copy(),
            f"{prefix}mlp.fc1.w": self.mlp.fc1.weight.detach().cpu().numpy().T.copy(),
            f"{prefix}mlp.fc1.b": self.mlp.fc1.bias.detach().cpu().numpy().copy(),
            f"{prefix}mlp.fc2.w": self.mlp.fc2.weight.detach().cpu().numpy().T.copy(),
            f"{prefix}mlp.fc2.b": self.mlp.fc2.bias.detach().cpu().numpy().copy(),
        }


class GPTModel(nn.Module):
    """Decoder-only transformer matching `ModelSpec` with tied LM head."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        self.wte = nn.Embedding(spec.vocab, spec.d_model)
        self.wpe = nn.Embedding(spec.seq_len, spec.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(spec) for _ in range(spec.n_layer)])
        self.ln_f = nn.LayerNorm(spec.d_model, eps=LN_EPS)
        pos = torch.arange(spec.seq_len, dtype=torch.long)
        self.register_buffer("pos_ids", pos, persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        proj_std = 0.02 / math.sqrt(2.0 * self.spec.n_layer)
        nn.init.normal_(self.wte.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.wpe.weight, mean=0.0, std=0.02)
        for block in self.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.normal_(block.attn.qkv.weight, mean=0.0, std=0.02)
            nn.init.zeros_(block.attn.qkv.bias)
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=proj_std)
            nn.init.zeros_(block.attn.proj.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)
            nn.init.normal_(block.mlp.fc1.weight, mean=0.0, std=0.02)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=proj_std)
            nn.init.zeros_(block.mlp.fc2.bias)
        nn.init.ones_(self.ln_f.weight)
        nn.init.zeros_(self.ln_f.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, seq_len = idx.shape
        x = self.wte(idx) + self.wpe(self.pos_ids[:seq_len])
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return F.linear(x, self.wte.weight)

    def load_numpy(self, params: dict[str, np.ndarray]) -> None:
        with torch.no_grad():
            self.wte.weight.copy_(torch.from_numpy(params["wte"]))
            self.wpe.weight.copy_(torch.from_numpy(params["wpe"]))
            for i, block in enumerate(self.blocks):
                block.load_numpy(params, prefix=f"blocks.{i}.")
            self.ln_f.weight.copy_(torch.from_numpy(params["ln_f.w"]))
            self.ln_f.bias.copy_(torch.from_numpy(params["ln_f.b"]))

    def to_numpy(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {
            "wte": self.wte.weight.detach().cpu().numpy().copy(),
            "wpe": self.wpe.weight.detach().cpu().numpy().copy(),
        }
        for i, block in enumerate(self.blocks):
            out.update(block.to_numpy(prefix=f"blocks.{i}."))
        out["ln_f.w"] = self.ln_f.weight.detach().cpu().numpy().copy()
        out["ln_f.b"] = self.ln_f.bias.detach().cpu().numpy().copy()
        return out
