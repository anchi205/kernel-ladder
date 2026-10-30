# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.53x faster
best      attempt #2, 3.45x (Load a 2D tile of 2 rows × 4096 columns per program so both rows' loads are issued before the reduction, which keeps more bytes in flight. The sum runs over axis=1, and I use 8 warps and drop the eviction hints.)

## tried
- #1 [speed] Each program handles several rows in a loop, so the weight vector is loaded once and stays in registers. The row sum-of-squares uses an in-register `tl.sum` with `rsqrt`, and x loads use an evict-first hint.: 3.35x (1.158 ms vs 3.880 ms)
- #2 [speed] Load a 2D tile of 2 rows × 4096 columns per program so both rows' loads are issued before the reduction, which keeps more bytes in flight. The sum runs over axis=1, and I use 8 warps and drop the eviction hints.: 3.45x (1.134 ms vs 3.915 ms)

## known
- Strategy for this step: Improve reductions: do row/tile reductions inside the kernel with cheap cooperative reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
