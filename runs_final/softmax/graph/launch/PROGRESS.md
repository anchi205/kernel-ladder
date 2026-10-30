# PROGRESS

goal      candidate for 'softmax' that matches the reference and runs >= 1.00x faster
best      attempt #1, 0.95x (Retune num_warps (4 warps for the 4096-wide row instead of 8, so each thread holds 32 elements and keeps more rows in flight per SM), with the algorithm unchanged.)

## tried
- #1 [speed] Retune num_warps (4 warps for the 4096-wide row instead of 8, so each thread holds 32 elements and keeps more rows in flight per SM), with the algorithm unchanged.: 0.95x (1.146 ms vs 1.094 ms)
- #2 [speed] Persistent grid-stride softmax that prefetches the next row's load into registers before computing the current row, to hide memory latency and close the gap to peak bandwidth.: 0.95x (1.145 ms vs 1.086 ms)

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
