"""PyTorch training step builder: eager / torch.compile, FP32 / FP16+GradScaler, and DDP."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from bench.model_torch import GPTModel
from bench.spec import (
    ADAMW_BETA1,
    ADAMW_BETA2,
    ADAMW_EPS,
    ADAMW_LR,
    ADAMW_WEIGHT_DECAY,
    LOSS_SCALE_BACKOFF_FACTOR,
    LOSS_SCALE_GROWTH_FACTOR,
    LOSS_SCALE_GROWTH_INTERVAL,
    LOSS_SCALE_INIT,
    ModelSpec,
)


@dataclass
class TorchStepHandle:
    """Container for a configured PyTorch training step and its components."""

    model: nn.Module
    raw_model: GPTModel
    optimizer: torch.optim.Optimizer
    scaler: torch.amp.GradScaler
    step_fn: Callable[[torch.Tensor], torch.Tensor]
    device: torch.device
    graph_breaks: int | None = None


def make_synthetic_pool(
    spec: ModelSpec,
    batch_size: int,
    *,
    device: torch.device,
    num_batches: int = 8,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Pre-generate synthetic token batches directly on the target device."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return [
        torch.randint(
            0,
            spec.vocab,
            (batch_size, spec.seq_len + 1),
            dtype=torch.long,
            device=device,
            generator=gen,
        )
        for _ in range(num_batches)
    ]


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_gb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)
    return 0.0


def count_graph_breaks(model: nn.Module, sample_tokens: torch.Tensor, precision: str) -> int | None:
    """Count TorchDynamo graph breaks on one forward pass, if available."""
    try:
        explanation = torch._dynamo.explain(
            lambda x: _forward_for_explain(model, x, precision),
        )(sample_tokens[:, :-1])
        return int(explanation.graph_break_count)
    except Exception:
        return None


def _forward_for_explain(model: nn.Module, x: torch.Tensor, precision: str) -> torch.Tensor:
    device_type = x.device.type
    with torch.amp.autocast(
        device_type=device_type,
        dtype=torch.float16,
        enabled=(precision == "fp16" and device_type == "cuda"),
    ):
        return model(x)


def init_distributed(backend: str = "nccl") -> tuple[int, int, int]:
    """Initialize torch.distributed from torchrun env vars; return (rank, local_rank, world)."""
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if backend == "nccl" and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def build_torch_step(
    spec: ModelSpec,
    *,
    device: torch.device,
    mode: str = "eager",
    precision: str = "fp16",
    distributed: bool = False,
    local_rank: int = 0,
    seed: int = 0,
    check_graph_breaks: bool = False,
) -> TorchStepHandle:
    """Build model, AdamW optimizer, GradScaler, optional compile/DDP, and step callable."""
    if mode not in ("eager", "compile"):
        raise ValueError(f"Unknown mode {mode!r}; expected 'eager' or 'compile'")
    if precision not in ("fp32", "fp16"):
        raise ValueError(f"Unknown precision {precision!r}; expected 'fp32' or 'fp16'")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    raw_model = GPTModel(spec).to(device)
    graph_breaks: int | None = None
    if mode == "compile" and check_graph_breaks:
        sample = torch.zeros((1, spec.seq_len + 1), dtype=torch.long, device=device)
        graph_breaks = count_graph_breaks(raw_model, sample, precision)

    model: nn.Module = raw_model
    if mode == "compile":
        model = torch.compile(model)  # type: ignore[assignment]
    if distributed:
        device_ids = [local_rank] if device.type == "cuda" else None
        model = DDP(model, device_ids=device_ids, gradient_as_bucket_view=True)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=ADAMW_LR,
        betas=(ADAMW_BETA1, ADAMW_BETA2),
        eps=ADAMW_EPS,
        weight_decay=ADAMW_WEIGHT_DECAY,
        fused=(device.type == "cuda"),
    )

    use_amp = precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=use_amp,
        init_scale=LOSS_SCALE_INIT,
        growth_factor=LOSS_SCALE_GROWTH_FACTOR,
        backoff_factor=LOSS_SCALE_BACKOFF_FACTOR,
        growth_interval=LOSS_SCALE_GROWTH_INTERVAL,
    )

    def step_fn(tokens: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(tokens[:, :-1])
            loss = F.cross_entropy(logits.float().flatten(0, 1), tokens[:, 1:].flatten())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss

    return TorchStepHandle(
        model=model,
        raw_model=raw_model,
        optimizer=optimizer,
        scaler=scaler,
        step_fn=step_fn,
        device=device,
        graph_breaks=graph_breaks,
    )
