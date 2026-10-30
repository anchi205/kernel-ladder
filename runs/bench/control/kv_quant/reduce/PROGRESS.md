# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.46x faster
best      attempt #1, 5.66x (Process several rows per program as a 2D [ROWS, BLOCK] tile, so each row's absmax is a cheap `tl.max(axis=1)` reduction. The tile stays in registers with vectorized loads and stores, and there are no atomics.)

## tried
- #1 [speed] Process several rows per program as a 2D [ROWS, BLOCK] tile, so each row's absmax is a cheap `tl.max(axis=1)` reduction. The tile stays in registers with vectorized loads and stores, and there are no atomics.: 5.66x (1.964 ms vs 11.111 ms)
- #2 [speed] Larger 2D tiles (32 rows × 128 cols per program, 8 warps) for more bytes in flight, int32 index math, and streaming `evict_first` loads, to push the memory-bound kernel closer to peak bandwidth.: 4.82x (2.306 ms vs 11.112 ms)

## known
- Strategy for this step: Improve reductions: do row/tile reductions inside the kernel with cheap cooperative reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
