# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.30x faster
best      attempt #1, 3.38x (Single-pass fused kernel. Each row is loaded into registers once, reduced with `tl.sum`, then normalized and stored from those registers. This removes the second read of x. Rows too wide for one tile fall back to the looped kernel.)

## tried
- #1 [ok] Single-pass fused kernel. Each row is loaded into registers once, reduced with `tl.sum`, then normalized and stored from those registers. This removes the second read of x. Rows too wide for one tile fall back to the looped kernel.: green

## known
- Strategy for this step: Improve reductions: do row/tile reductions inside the kernel with cheap cooperative reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
