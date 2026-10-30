# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.53x faster
best      attempt #1, 3.44x (Autotune `num_warps` (2/4/8/16) per row width N with the algorithm unchanged, instead of the fixed 4-or-8 heuristic.)

## tried
- #1 [speed] Autotune `num_warps` (2/4/8/16) per row width N with the algorithm unchanged, instead of the fixed 4-or-8 heuristic.: 3.44x (1.121 ms vs 3.859 ms)
- #2 [speed] Add rows-per-program (1/2/4) to the autotune grid so the weight vector is loaded once per program, with evict_first loads for x and streaming (.cs) stores, so the streaming traffic doesn't pollute L2. The kernel was already at 92% of peak bandwidth, so I'm only tuning launch and caching parameters.: 3.35x (1.138 ms vs 3.811 ms)

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
