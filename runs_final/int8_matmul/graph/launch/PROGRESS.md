# PROGRESS

goal      candidate for 'int8_matmul' that matches the reference and runs >= 7.41x faster
best      attempt #1, 7.42x (Tune launch parameters only: widen the autotune search (tile shapes, warps, stages, GROUP_M) and use a larger tile in the B-transpose kernel. The algorithm is unchanged.)

## tried
- #1 [ok] Tune launch parameters only: widen the autotune search (tile shapes, warps, stages, GROUP_M) and use a larger tile in the B-transpose kernel. The algorithm is unchanged.: green

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
