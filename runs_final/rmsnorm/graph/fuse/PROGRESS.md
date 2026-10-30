# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.53x faster
best      attempt #2, 3.43x (Process two rows per program, with both rows' loads issued before any reduction, to put more bytes in flight toward the ~95% of peak bandwidth needed. This uses 8 warps and drops the eviction hints on the loads.)

## tried
- #1 [speed] One fused kernel (load, sum of squares, rsqrt, scale by w, store) with `rsqrt` instead of `1/sqrt`, `evict_first` hints on the streamed x loads and out stores, and fewer warps per row for N up to 4096 to cut reduction overhead.: 3.33x (1.143 ms vs 3.806 ms)
- #2 [speed] Process two rows per program, with both rows' loads issued before any reduction, to put more bytes in flight toward the ~95% of peak bandwidth needed. This uses 8 warps and drops the eviction hints on the loads.: 3.43x (1.130 ms vs 3.872 ms)

## known
- Strategy for this step: Fuse separate kernels and intermediate tensors into as few kernels as possible, so intermediate results never round-trip through global memory.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
