# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.54x faster
best      attempt #2, 3.39x (Add cache-eviction hints to the single-pass row kernel (streaming `x` and `out` with `evict_first`, `w` with `evict_last`) and keep `num_warps=4` for N=4096. The kernel is already at 98% of peak DRAM bandwidth, so this is a small bet to keep the shared 16 KB `w` resident in L2.)

## tried
- #1 [speed] Launch-only tuning: for the N=4096 row kernel, drop num_warps from 8 to 4 so each thread holds 32 floats. That gives more loads in flight per thread and lets more rows run concurrently per SM. The algorithm is unchanged.: 3.37x (1.129 ms vs 3.803 ms)
- #2 [speed] Add cache-eviction hints to the single-pass row kernel (streaming `x` and `out` with `evict_first`, `w` with `evict_last`) and keep `num_warps=4` for N=4096. The kernel is already at 98% of peak DRAM bandwidth, so this is a small bet to keep the shared 16 KB `w` resident in L2.: 3.39x (1.141 ms vs 3.866 ms)

## known
- Strategy for this step: Tune launch parameters only: block/tile sizes, num_warps, num_stages, grid shape. Do not change the algorithm.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
