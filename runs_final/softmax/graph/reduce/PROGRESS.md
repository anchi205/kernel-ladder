# PROGRESS

goal      candidate for 'softmax' that matches the reference and runs >= 0.98x faster
best      attempt #2, 0.98x (Use 8 warps per row with an unmasked, fully vectorized path when the row width is a power of two. Loads use default caching and stores use a streaming hint (`.cs`). This replaces the earlier `evict_first` load hint and 4-warp config.)

## tried
- #1 [speed] Keep the whole row in registers and use cooperative `tl.max`/`tl.sum` reductions with `exp2` and a single reciprocal multiply. The num_warps choice (4 for 4096 columns) is an untested guess.: 0.95x (1.138 ms vs 1.081 ms)
- #2 [ok] Use 8 warps per row with an unmasked, fully vectorized path when the row width is a power of two. Loads use default caching and stores use a streaming hint (`.cs`). This replaces the earlier `evict_first` load hint and 4-warp config.: green

## known
- Strategy for this step: Improve reductions: do row/tile reductions inside the kernel with cheap cooperative reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
