"""Canonical model specification, presets, FLOP/parameter formulas, and NumPy init."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

# Fixed optimizer and training hyperparameters shared by PyTorch and JAX.
ADAMW_LR: float = 3e-4
ADAMW_BETA1: float = 0.9
ADAMW_BETA2: float = 0.95
ADAMW_EPS: float = 1e-8
ADAMW_WEIGHT_DECAY: float = 0.1
LN_EPS: float = 1e-5

# Dynamic FP16 loss scaling constants (matched to PyTorch GradScaler defaults).
LOSS_SCALE_INIT: float = 65536.0
LOSS_SCALE_GROWTH_FACTOR: float = 2.0
LOSS_SCALE_BACKOFF_FACTOR: float = 0.5
LOSS_SCALE_GROWTH_INTERVAL: int = 2000

# Hardware reference ceilings for NVIDIA T4 (sm_75).
T4_FP16_PEAK_TFLOPS: float = 65.0
T4_FP32_PEAK_TFLOPS: float = 8.1
T4_MEM_BANDWIDTH_GBS: float = 320.0
T4_EXPECTED_GPU_NAME: str = "Tesla T4"
T4_EXPECTED_CC: str = "7.5"
T4_EXPECTED_VRAM_MIB: int = 15360


@dataclass(frozen=True)
class ModelSpec:
    """Decoder-only GPT transformer configuration with tied LM head."""

    n_layer: int = 24
    d_model: int = 1024
    n_head: int = 16
    seq_len: int = 512
    vocab: int = 1024
    mlp_ratio: int = 4

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def mlp_dim(self) -> int:
        return self.mlp_ratio * self.d_model

    def n_params_per_block(self) -> int:
        d = self.d_model
        m = self.mlp_dim
        ln1 = 2 * d
        qkv = d * 3 * d + 3 * d
        proj = d * d + d
        ln2 = 2 * d
        fc1 = d * m + m
        fc2 = m * d + d
        return ln1 + qkv + proj + ln2 + fc1 + fc2

    def n_params(self) -> int:
        """Exact trainable parameter count (tied token embedding and LM head)."""
        d = self.d_model
        embed = self.vocab * d + self.seq_len * d
        ln_f = 2 * d
        return self.n_layer * self.n_params_per_block() + embed + ln_f

    @property
    def grad_bytes_fp32(self) -> int:
        """Total FP32 gradient buffer size in bytes for one replica."""
        return 4 * self.n_params()

    def train_flops_per_step(self, global_batch: int) -> float:
        """Approximate forward + backward matmul FLOPs for one training step."""
        tokens = global_batch * self.seq_len
        linear_coeff = 4 + 2 * self.mlp_ratio  # 12 when mlp_ratio == 4
        n_matmul = self.n_layer * linear_coeff * (self.d_model**2) + self.vocab * self.d_model
        attn = 12 * self.n_layer * self.seq_len * self.d_model
        return float(tokens * (6 * n_matmul + attn))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


PRESETS: dict[str, ModelSpec] = {
    "medium": ModelSpec(),
    "small": ModelSpec(n_layer=12, d_model=768, n_head=12),
    "tiny": ModelSpec(n_layer=2, d_model=64, n_head=4, seq_len=32, vocab=128),
}


def get_spec(preset: str) -> ModelSpec:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; expected one of {sorted(PRESETS)}")
    return PRESETS[preset]


def init_block_numpy_params(
    spec: ModelSpec,
    rng: np.random.Generator,
    prefix: str = "",
) -> dict[str, np.ndarray]:
    """Create GPT-2-style FP32 weights for one transformer block in canonical (in, out) layout."""
    d = spec.d_model
    m = spec.mlp_dim
    std = 0.02
    proj_std = std / np.sqrt(2.0 * spec.n_layer)

    def norm(shape: tuple[int, ...], s: float) -> np.ndarray:
        return rng.normal(0.0, s, size=shape).astype(np.float32)

    return {
        f"{prefix}ln1.w": np.ones((d,), dtype=np.float32),
        f"{prefix}ln1.b": np.zeros((d,), dtype=np.float32),
        f"{prefix}attn.qkv.w": norm((d, 3 * d), std),
        f"{prefix}attn.qkv.b": np.zeros((3 * d,), dtype=np.float32),
        f"{prefix}attn.proj.w": norm((d, d), proj_std),
        f"{prefix}attn.proj.b": np.zeros((d,), dtype=np.float32),
        f"{prefix}ln2.w": np.ones((d,), dtype=np.float32),
        f"{prefix}ln2.b": np.zeros((d,), dtype=np.float32),
        f"{prefix}mlp.fc1.w": norm((d, m), std),
        f"{prefix}mlp.fc1.b": np.zeros((m,), dtype=np.float32),
        f"{prefix}mlp.fc2.w": norm((m, d), proj_std),
        f"{prefix}mlp.fc2.b": np.zeros((d,), dtype=np.float32),
    }


def init_numpy_params(spec: ModelSpec, seed: int = 0) -> dict[str, np.ndarray]:
    """Create full GPT model FP32 weights in canonical NumPy format."""
    rng = np.random.default_rng(seed)
    d = spec.d_model
    params: dict[str, np.ndarray] = {
        "wte": rng.normal(0.0, 0.02, size=(spec.vocab, d)).astype(np.float32),
        "wpe": rng.normal(0.0, 0.02, size=(spec.seq_len, d)).astype(np.float32),
    }
    for i in range(spec.n_layer):
        params.update(init_block_numpy_params(spec, rng, prefix=f"blocks.{i}."))
    params["ln_f.w"] = np.ones((d,), dtype=np.float32)
    params["ln_f.b"] = np.zeros((d,), dtype=np.float32)
    return params
