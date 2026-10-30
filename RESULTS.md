# KernelLadder results (GB10 / sm_121, one machine, one run per cell)

An agent (Claude via `claude -p`) writes a Triton kernel for each op; a fixed verifier checks correctness (float32 and float16
shapes, edge cases, determinism, no input mutation, a fence against calling the op's own library function) and times it; a strategy graph
(seed -> fuse / reduce / vectorize / tile / launch, merge, gate) improves it. Weights/inputs are random; timings are medians. Winners below are from `runs_final/<op>/graph/final.py`.

## Speedup, fp32 (fp16 in brackets). `>1` means the agent's kernel is faster.

| op | vs naive reference | vs torch.compile(reference) | vs strongest library op | verdict |
|---|---|---|---|---|
| softmax | 0.98x (1.03x) | 1.01x (1.00x) | reference IS torch's fused softmax: 0.98x | **no gain**: torch is already at the memory ceiling |
| rmsnorm | 3.41x (9.54x) | 0.99x (0.99x) | F.rms_norm: 1.20x (1.00x) | **ties torch.compile**; 20% over F.rms_norm in fp32 only |
| kv_quant (2 outputs) | 8.39x (15.5x) | 1.00x (1.01x) | no fused library op | **ties torch.compile** |
| int8_matmul (scaled) | 7.56x (14.1x) | 6.14x (9.44x) | torch._int_mm + scales: 5.81x (11.4x) | **clear win** (fused scale epilogue) |
| attention (causal) | 7.29x (54x) | 3.48x (13.5x) | torch SDPA: **0.41x** (0.91x) | **loses** to PyTorch's fused attention |

How to read it: against the naive eager reference the agent looks 3-8x faster, but for rmsnorm, kv_quant and softmax that is exactly what
`torch.compile` gives for free; the agent re-derives operator fusion. The one op where it beat every library alternative is int8_matmul
(note the `_int_mm` baseline is `_int_mm` followed by two unfused scale multiplies, i.e. a composite, not one fused library kernel).
Attention lost to SDPA in both precisions in this run. An earlier attention run produced a different kernel (8.5x vs naive, 1.32x faster than SDPA in
fp16 once), so the agent's output varies a lot run to run; one run per cell says little about its distribution.

## What the graph did
* softmax / rmsnorm / kv_quant: the first correct kernel was already at the memory ceiling (rmsnorm: 242 GB/s = 92% of a 264 GB/s streaming-read
  ceiling), so the four strategy nodes gained nothing and spent their attempts anyway (8 wasted model attempts per op).
* int8_matmul: the strategies chained as designed: seed 5.46x -> vectorize 6.73x -> tile 7.27x -> launch 7.42x.
* attention: seed 6.53x -> fuse 7.14x; tile / reduce / launch did not beat it.
* Cost for the five ops: 21 min wall, 11 min of GPU time (48 / 64 / 61 / 388 / 97 s).

## Not shown / caveats
* **Learning across runs is not demonstrated** (one earlier benchmark, kv_quant 8.3x -> 8.2x, one run per arm). Given the above, the lessons store should
  let later runs skip fruitless strategies; that is a hypothesis, not a result.
* One run per cell; the agent is nondeterministic; fp32 inputs; random data; tolerance is tight in fp32.
* The gate's reviewer is the same model family as the code writer. The gate previously shipped a 0.98x softmax kernel; it now holds anything not faster
  than the reference (fixed after this run, tested, not re-run on the GPU).
* No raw CUDA (Triton / PyTorch only), no KernelBench run, API-based proposer untested (no key), ncu/compute-sanitizer unavailable on this GPU
  (determinism check and torch.profiler stand in).
* The profiler's peak was changed from a copy-based 228 GB/s to a streaming-read 264 GB/s ceiling before these runs.

## What you can and cannot reproduce

* The checker is deterministic: the same candidate gets the same verdict (the CPU-side tests run anywhere).
* The agent is not: re-running `graph` gives different kernels and different speedups.
* Every speedup above was measured on one NVIDIA GB10 (sm_121) with torch 2.14.1+cu130 and Triton 3.8.0, one run per cell.
  Other GPUs will give different numbers; the bandwidth-ceiling percentages are specific to this machine.
