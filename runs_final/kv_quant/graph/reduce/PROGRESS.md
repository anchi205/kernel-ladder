# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.58x faster
best      attempt #1, 8.09x (Process several rows per program as a 2D tile [ROWS, BLOCK]. The per-row absmax is an in-register `tl.max(axis=1)` over vectorized loads, so there is one launch with fewer programs and no atomics. The exact `div_rn` and `rint` math is unchanged.)

## tried
- #1 [speed] Process several rows per program as a 2D tile [ROWS, BLOCK]. The per-row absmax is an in-register `tl.max(axis=1)` over vectorized loads, so there is one launch with fewer programs and no atomics. The exact `div_rn` and `rint` math is unchanged.: 8.09x (1.373 ms vs 11.104 ms)
- #2 [speed] Same 2D-tile kernel at 93% of peak, but with 16 rows per program (16 elements per thread) for more loads in flight, plus streaming cache hints (`evict_first` on loads, `.cs` on stores).: 5.45x (2.037 ms vs 11.110 ms)

## known
- Strategy for this step: Improve reductions: do row/tile reductions inside the kernel with cheap cooperative reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
