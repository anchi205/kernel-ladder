# PROGRESS

goal      candidate for 'attention' that matches the reference and runs >= 7.29x faster
best      attempt #1, 4.51x (Launch-parameter tuning only: BLOCK_N 32→64, so each K/V iteration does more work and the loop runs half as many times, with BLOCK_M=64, num_warps=4 and num_stages=2 kept. The algorithm is unchanged.)

## tried
- #1 [speed] Launch-parameter tuning only: BLOCK_N 32→64, so each K/V iteration does more work and the loop runs half as many times, with BLOCK_M=64, num_warps=4 and num_stages=2 kept. The algorithm is unchanged.: 4.51x (2.896 ms vs 13.070 ms)
- #2 [correctness] Switch both `tl.dot` calls from `input_precision="ieee"` (scalar FMA) to `"tf32x3"` (3-pass tensor-core, near-fp32 accuracy), with BLOCK_M=128, BLOCK_N=64 and 8 warps.: OutOfResources(114688, 101376, 'shared memory')

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
