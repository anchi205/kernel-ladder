# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.38x faster
best      attempt #1, 8.01x (Vectorize with wider tiles (8 rows per program, 4 warps), mask-free loads and stores when D is a power of two, and streaming `evict_first` loads. This should give 16B coalesced fp32 loads and wider int8 stores while keeping the exact rounding.)

## tried
- #1 [speed] Vectorize with wider tiles (8 rows per program, 4 warps), mask-free loads and stores when D is a power of two, and streaming `evict_first` loads. This should give 16B coalesced fp32 loads and wider int8 stores while keeping the exact rounding.: 8.01x (1.387 ms vs 11.108 ms)
- #2 [speed] Persistent grid-stride kernel with `tl.range` software pipelining (num_stages=3) so more loads are in flight, plus `evict_first` on stores. The previous kernel was already at about 99% of the nominal bandwidth peak, so the remaining headroom is in memory-level parallelism. Rounding is unchanged.: 7.83x (1.418 ms vs 11.105 ms)

## known
- Strategy for this step: Make global-memory access efficient: coalesced, aligned, vectorized loads and stores; avoid redundant reads.
- Lesson (sm_121, memory-bound): 'fuse' gained on 1 of 1 op(s), up to 1.07x.
- Lesson (sm_121, memory-bound): 'vectorize' gained on 1 of 1 op(s), up to 1.05x.
- Lesson (sm_121, memory-bound): 'reduce' gained on 1 of 1 op(s), up to 1.04x.
- Lesson (sm_121, memory-bound): 'launch' gained nothing on 1 op(s) tried.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
