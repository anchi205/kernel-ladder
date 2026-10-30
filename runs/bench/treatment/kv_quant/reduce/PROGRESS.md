# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.38x faster
best      attempt #1, 8.04x (Reduce each row with a cooperative in-register warp-shuffle `tl.max` (one row per warp, 8 rows per program), and replace the multi-op round-half-even sequence with the fp32 magic-number trick (y + 1.5·2²³ − 1.5·2²³).)

## tried
- #1 [speed] Reduce each row with a cooperative in-register warp-shuffle `tl.max` (one row per warp, 8 rows per program), and replace the multi-op round-half-even sequence with the fp32 magic-number trick (y + 1.5·2²³ − 1.5·2²³).: 8.04x (1.383 ms vs 11.110 ms)
- #2 [correctness] Raise bytes in flight per thread (16 rows per program) and mark the streaming load `evict_first`/`.cg` so the 268 MB read doesn't pollute L2, to close the last few percent to the bandwidth ceiling.: PTXASError("Internal Triton PTX codegen error\n`ptxas` stderr:\nptxas-blackwell /tmp/tmplh38vsnm.ptx, line 74; error   : Modifier '.evict_first' cannot be combi

## known
- Strategy for this step: Improve reductions: do row/tile reductions inside the kernel with cheap cooperative reductions (e.g. tl.sum / warp-level), keep partial sums in registers, avoid atomics.
- Lesson (sm_121, memory-bound): 'fuse' gained on 1 of 1 op(s), up to 1.07x.
- Lesson (sm_121, memory-bound): 'vectorize' gained on 1 of 1 op(s), up to 1.05x.
- Lesson (sm_121, memory-bound): 'reduce' gained on 1 of 1 op(s), up to 1.04x.
- Lesson (sm_121, memory-bound): 'launch' gained nothing on 1 op(s) tried.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
