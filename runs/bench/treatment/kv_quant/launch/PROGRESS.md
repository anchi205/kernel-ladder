# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.38x faster
best      attempt #1, 8.04x (Launch-parameter tuning only. I raised ROWS from 4 to 8 per program and set num_warps=4 explicitly, so each thread handles 8 elements (two 16-byte loads) and has more bytes in flight. The algorithm is unchanged.)

## tried
- #1 [speed] Launch-parameter tuning only. I raised ROWS from 4 to 8 per program and set num_warps=4 explicitly, so each thread handles 8 elements (two 16-byte loads) and has more bytes in flight. The algorithm is unchanged.: 8.04x (1.382 ms vs 11.105 ms)

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.
- Lesson (sm_121, memory-bound): 'fuse' gained on 1 of 1 op(s), up to 1.07x.
- Lesson (sm_121, memory-bound): 'vectorize' gained on 1 of 1 op(s), up to 1.05x.
- Lesson (sm_121, memory-bound): 'reduce' gained on 1 of 1 op(s), up to 1.04x.
- Lesson (sm_121, memory-bound): 'launch' gained nothing on 1 op(s) tried.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
