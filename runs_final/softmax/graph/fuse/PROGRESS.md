# PROGRESS

goal      candidate for 'softmax' that matches the reference and runs >= 0.98x faster
best      attempt #1, 0.95x (Single fused kernel (load, max, exp2, sum, normalize, store in one pass, one HBM read and one write), with exp2 and a reciprocal multiply instead of a divide, and fewer warps per row to cut reduction overhead.)

## tried
- #1 [speed] Single fused kernel (load, max, exp2, sum, normalize, store in one pass, one HBM read and one write), with exp2 and a reciprocal multiply instead of a divide, and fewer warps per row to cut reduction overhead.: 0.95x (1.132 ms vs 1.078 ms)
- #2 [speed] Persistent grid-stride softmax kernel with software-pipelined row loads (`num_stages=2`), exp2, and no cache or eviction hints, so the next row's HBM read overlaps the current row's reduction and store.: 0.92x (1.169 ms vs 1.077 ms)

## known
- Strategy for this step: Fuse separate kernels and intermediate tensors into as few kernels as possible, so intermediate results never round-trip through global memory.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
