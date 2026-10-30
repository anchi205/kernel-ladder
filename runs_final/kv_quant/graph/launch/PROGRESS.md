# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.58x faster
best      attempt #1, 8.25x (Launch-shape tuning: each program handles several rows as a 2D tile (about 512 elements) with 4 warps, so each warp works on one row with 16B vector loads. This replaces one row per program on 1 warp and cuts the program count roughly 4x.)

## tried
- #1 [speed] Launch-shape tuning: each program handles several rows as a 2D tile (about 512 elements) with 4 warps, so each warp works on one row with 16B vector loads. This replaces one row per program on 1 warp and cuts the program count roughly 4x.: 8.25x (1.347 ms vs 11.108 ms)
- #2 [speed] Launch-shape tuning: larger tiles (16 rows, about 2048 elements) at 4 warps, so each thread keeps four 16B loads in flight, plus an evict_first hint on the streaming load.: 5.42x (2.050 ms vs 11.109 ms)

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
