# Two NVIDIA T4s: PyTorch vs JAX Data-Parallel Scaling

> **Once a training step is large enough to use a T4, what does the second T4 return, in PyTorch and in JAX?**

<!-- results:start -->
_Pending first 2× NVIDIA T4 run. Run `bash scripts/setup.sh && bash scripts/run_all.sh` and regenerate with `python -m bench.report`._
<!-- results:end -->

## Predictions (committed before any T4 run)

1. **FP16 beats FP32 on one T4 by a wide margin (> 1.5×)** once tensor cores are engaged at the frozen workload size.
2. **On one T4, JAX `jit` and the chosen PyTorch FP16 mode land close.** Any gap larger than run-to-run step spread gets a profiler trace rather than an unverified claim.
3. **Weak scaling lands under 2× by roughly the FP32 allreduce sync time** measured in Experiment 1 (`~1.22 GB / β`). Strong scaling efficiency lands lower than weak scaling because compute per card halves while gradient sync stays fixed.

## Workload & Fairness Controls

- **Model:** GPT-2-medium body (24 layers, `d_model=1024`, 16 heads, `seq_len=512`, `vocab=1024`, tied LM head) = **303,884,288** parameters (~1.22 GB FP32 gradient sync).
- **Same math on both stacks:** explicit causal attention (`einsum` + FP32 softmax), tanh-GELU, no dropout, mirrored autocast policy (FP16 matmuls; FP32 LayerNorm/softmax/loss/master weights), dynamic FP16 loss scaling in both PyTorch (`GradScaler`) and JAX (`DynamicLossScale`), and identical AdamW (`lr=3e-4`, `β=(0.9, 0.95)`, `wd=0.1`).
- **Expected asymmetry:** PyTorch DDP uses one process per GPU (`torchrun`), while JAX uses a single controller process with `jax.sharding` across both GPUs.
- **Timing boundary:** synthetic token batches are pre-allocated on GPU; CPU cores and disk I/O stay outside the timed step.

## Reproduce on 2× NVIDIA T4

```bash
git clone https://github.com/adityavsb/pytorch-jax-gpt-training.git
cd pytorch-jax-gpt-training
bash scripts/setup.sh
bash scripts/run_all.sh
bash scripts/package_results.sh   # writes results-<date>.tar.gz
```

## Local CPU Tests

```bash
pytest -q
bash scripts/smoke_cpu.sh
```
