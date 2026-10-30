# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.46x faster
best      attempt #1, 8.04x (Launch-only tuning: each program handles several rows (about 1024 elements, a 2D [ROWS, BLOCK_D] tile) with 4 warps, so each thread does vectorized loads and stores and the grid is smaller. The algorithm is unchanged.)

## tried
- #1 [speed] Launch-only tuning: each program handles several rows (about 1024 elements, a 2D [ROWS, BLOCK_D] tile) with 4 warps, so each thread does vectorized loads and stores and the grid is smaller. The algorithm is unchanged.: 8.04x (1.381 ms vs 11.104 ms)
- #2 [correctness] Keep the single-pass kernel, which is already at about 99% of the measured bandwidth. Add streaming cache hints (`.cg` and `evict_first` on the loads, `evict_first` on the stores) so the one-shot 268 MB read and the int8 write don't pollute L1/L2. Use a slightly larger tile of 2048 elements with 8 warps to keep more loads in flight.: PTXASError("Internal Triton PTX codegen error\n`ptxas` stderr:\nptxas-blackwell /tmp/tmpq2sv94t5.ptx, line 117; error   : Modifier '.evict_first' cannot be comb

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
