"""CPU test verifying 2-process PyTorch DDP (gloo) equals single-process step."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp

from bench.spec import get_spec, init_numpy_params
from bench.step_torch import build_torch_step, cleanup_distributed


def _ddp_worker(rank: int, init_file: str, out_dir: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29507"
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        spec = get_spec("tiny")
        w_np = init_numpy_params(spec, seed=19)
        tokens_all = np.load(Path(out_dir) / "tokens.npy")
        local_tokens = torch.from_numpy(tokens_all[rank * 2 : (rank + 1) * 2])

        handle = build_torch_step(
            spec,
            device=torch.device("cpu"),
            mode="eager",
            precision="fp32",
            distributed=True,
            local_rank=rank,
            seed=19,
        )
        handle.raw_model.load_numpy(w_np)
        loss = handle.step_fn(local_tokens)
        loss_avg = loss.detach().clone()
        dist.all_reduce(loss_avg, op=dist.ReduceOp.SUM)
        loss_avg /= 2.0
        if rank == 0:
            np.savez(Path(out_dir) / "ddp_params.npz", **handle.raw_model.to_numpy())
            np.save(Path(out_dir) / "ddp_loss.npy", np.array([float(loss_avg)]))
    finally:
        cleanup_distributed()


def test_torch_ddp_matches_single_process() -> None:
    spec = get_spec("tiny")
    w_np = init_numpy_params(spec, seed=19)
    rng = np.random.default_rng(23)
    tokens_all = rng.integers(0, spec.vocab, size=(4, spec.seq_len + 1), dtype=np.int64)

    # Reference single-process step on full batch=4
    ref = build_torch_step(
        spec,
        device=torch.device("cpu"),
        mode="eager",
        precision="fp32",
        distributed=False,
        seed=19,
    )
    ref.raw_model.load_numpy(w_np)
    ref_loss = float(ref.step_fn(torch.from_numpy(tokens_all)).detach())
    ref_params = ref.raw_model.to_numpy()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        np.save(tmp_path / "tokens.npy", tokens_all)
        init_file = str(tmp_path / "dist_init")
        mp.spawn(_ddp_worker, args=(init_file, tmp), nprocs=2, join=True)

        ddp_loss = float(np.load(tmp_path / "ddp_loss.npy")[0])
        ddp_params = dict(np.load(tmp_path / "ddp_params.npz"))

    assert abs(ref_loss - ddp_loss) < 1e-5, f"DDP loss mismatch: {abs(ref_loss - ddp_loss)}"
    for k in ref_params:
        diff = float(np.max(np.abs(ref_params[k] - ddp_params[k])))
        assert diff < 1e-5, f"DDP param {k} mismatch: {diff}"
