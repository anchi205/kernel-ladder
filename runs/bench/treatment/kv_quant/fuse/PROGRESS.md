# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.38x faster
best      attempt #2, 8.09x (The kernel is already at about 99% of peak DRAM bandwidth, so this attempt only changes how much is in flight. I use 16 rows per program with 8 warps for more loads in flight per SM, and add `evict_first` hints on the int8 and scale stores. The byte count is fixed, so this may gain little.)

## tried
- #1 [speed] Keep the single fused kernel and replace the multi-op manual round-half-even with one `libdevice.rint`. Also add an `evict_first` hint on the streaming loads, use 8 rows per program, and set `num_warps=4`.: 8.07x (1.378 ms vs 11.121 ms)
- #2 [speed] The kernel is already at about 99% of peak DRAM bandwidth, so this attempt only changes how much is in flight. I use 16 rows per program with 8 warps for more loads in flight per SM, and add `evict_first` hints on the int8 and scale stores. The byte count is fixed, so this may gain little.: 8.09x (1.376 ms vs 11.122 ms)

## known
- Strategy for this step: Fuse separate kernels and intermediate tensors into as few kernels as possible, so intermediate results never round-trip through global memory.
- Lesson (sm_121, memory-bound): 'fuse' gained on 1 of 1 op(s), up to 1.07x.
- Lesson (sm_121, memory-bound): 'vectorize' gained on 1 of 1 op(s), up to 1.05x.
- Lesson (sm_121, memory-bound): 'reduce' gained on 1 of 1 op(s), up to 1.04x.
- Lesson (sm_121, memory-bound): 'launch' gained nothing on 1 op(s) tried.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
