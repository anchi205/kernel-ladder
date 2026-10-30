# PROGRESS

goal      candidate for 'attention' that matches the reference and runs >= 6.66x faster
best      attempt #2, 5.56x (Keep the earlier design (per-thread partial row-sum tile reduced once after the loop, exp2 with the scale folded in, unmasked fully-visible blocks), and fix the compile error by casting `p` to V's dtype before the PV `tl.dot`.)

## tried
- #1 [correctness] Keep the row-sum as a per-thread [M,N] partial tile that is reduced once after the loop instead of every iteration, use exp2 with the scale folded in, and run unmasked loads and no mask math for blocks fully below the diagonal.: CompilationError('def _attn_fwd(\n    Q, K, V, O,\n    N, qk_scale,\n    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,\n):\n    pid_m = tl.prog
- #2 [speed] Keep the earlier design (per-thread partial row-sum tile reduced once after the loop, exp2 with the scale folded in, unmasked fully-visible blocks), and fix the compile error by casting `p` to V's dtype before the PV `tl.dot`.: 5.56x (2.365 ms vs 13.153 ms)

## known
- Strategy for this step: Improve reductions: do row/tile reductions inside the kernel with cheap cooperative reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
