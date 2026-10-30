# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.30x faster
best      attempt #1, 3.47x (Fuse the two passes into one single-load kernel. Each row is read from global memory once into registers, then the sum of squares, rsqrt and scale-by-weight happen on it before one store. Rows too wide for registers fall back to the loop kernel.)

## tried
- #1 [ok] Fuse the two passes into one single-load kernel. Each row is read from global memory once into registers, then the sum of squares, rsqrt and scale-by-weight happen on it before one store. Rows too wide for registers fall back to the loop kernel.: green

## known
- Strategy for this step: Fuse separate kernels and intermediate tensors into as few kernels as possible, so intermediate results never round-trip through global memory.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
