# KernelLadder benchmark (sm_121)

train ops: rmsnorm | held-out ops: kv_quant

| op | family | arm | speedup | re-measured | model attempts | skipped | wall s | GPU s | gate |
|---|---|---|---|---|---|---|---|---|---|
| rmsnorm | memory-bound | train | 3.47x | 3.47x | 8 | 0 | 168 | 42 | ship |
| kv_quant | memory-bound | control | 8.3x | 8.21x | 9 | 0 | 146 | 44 | hold |
| kv_quant | memory-bound | treatment | 8.22x | 8.28x | 8 | 0 | 125 | 41 | ship |

## Held-out: with lessons vs without

- **kv_quant**: speedup 8.3x -> 8.22x, model attempts 9 -> 8, GPU time 44s -> 41s, wall 146s -> 125s

> Each cell is **one run**. The proposer is nondeterministic and the GPU is shared, so differences
> of a few percent in speedup or wall time are noise; attempt counts are the least noisy signal of
> search cost. Treat this as a worked example of the method, not a statistic.