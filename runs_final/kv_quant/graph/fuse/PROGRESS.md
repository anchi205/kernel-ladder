# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.58x faster
best      attempt #1, 8.09x (Keep the single fused kernel (load, absmax, scale, divide, round, clamp, int8 store, scale store) and have each program handle several rows as a 2D tile. This gives wider, better-coalesced accesses and fewer, fatter programs, so more bytes are in flight per SM.)

## tried
- #1 [speed] Keep the single fused kernel (load, absmax, scale, divide, round, clamp, int8 store, scale store) and have each program handle several rows as a 2D tile. This gives wider, better-coalesced accesses and fewer, fatter programs, so more bytes are in flight per SM.: 8.09x (1.375 ms vs 11.121 ms)
- #2 [speed] Keep the single fused kernel, which is already at 92% of peak bandwidth. Make each program handle 16 rows instead of 8, so each thread has more 16-byte loads in flight. Also mark the streaming load and store `evict_first` so they don't pollute L2.: 5.47x (2.033 ms vs 11.113 ms)

## known
- Strategy for this step: Fuse separate kernels and intermediate tensors into as few kernels as possible, so intermediate results never round-trip through global memory.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
