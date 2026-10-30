# PROGRESS

goal      candidate for 'int8_matmul' that matches the reference and runs >= 6.86x faster
best      attempt #2, 7.27x (Replace the slow `b.t().contiguous()` elementwise kernel (85us of the 448us per call) with a tiled Triton transpose kernel that coalesces both the read and the write. The GEMM kernel is unchanged.)

## tried
- #1 [speed] Tiled int8 tensor-core GEMM with grouped (L2-friendly) tile ordering, a K-contiguous pre-transposed B, and a wider autotune sweep over tile shapes, K-depth and stage counts that stay within the ~99KB shared-memory limit.: 6.84x (0.454 ms vs 3.108 ms)
- #2 [ok] Replace the slow `b.t().contiguous()` elementwise kernel (85us of the 448us per call) with a tiled Triton transpose kernel that coalesces both the read and the write. The GEMM kernel is unchanged.: green

## known
- Strategy for this step: Tile the computation so each block reuses data from registers or shared memory instead of re-reading global memory; pick tile shapes that fit the hardware.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
