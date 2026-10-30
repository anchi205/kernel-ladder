# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.46x faster
best      attempt #2, 8.09x (Same fused kernel, now with streaming cache hints (`evict_first` on the load, `.cs` on the stores) and a fatter tile of 16 rows with 8 warps, to squeeze the last bit of DRAM bandwidth. The profile shows the kernel already at 99% of peak, so this is a small gain at best.)

## tried
- #1 [speed] One fused kernel (absmax, scale, divide, round, clamp, int8 store) that handles 8 rows per program as a 2D tile. This gives wider vectorized loads and fewer, fatter programs than one row per program, so the memory-bound pass gets closer to peak bandwidth.: 8.01x (1.387 ms vs 11.106 ms)
- #2 [speed] Same fused kernel, now with streaming cache hints (`evict_first` on the load, `.cs` on the stores) and a fatter tile of 16 rows with 8 warps, to squeeze the last bit of DRAM bandwidth. The profile shows the kernel already at 99% of peak, so this is a small gain at best.: 8.09x (1.375 ms vs 11.128 ms)

## known
- Strategy for this step: Fuse separate kernels and intermediate tensors into as few kernels as possible, so intermediate results never round-trip through global memory.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
