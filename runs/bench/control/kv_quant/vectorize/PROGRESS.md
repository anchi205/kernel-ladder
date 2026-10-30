# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.46x faster
best      attempt #1, 8.07x (Process several rows per program as a 2D [ROWS, BLOCK] tile so each thread handles contiguous elements. That lets Triton emit 16-byte vectorized fp32 loads and 4-byte int8 stores, with the per-row absmax reduced along axis 1.)

## tried
- #1 [speed] Process several rows per program as a 2D [ROWS, BLOCK] tile so each thread handles contiguous elements. That lets Triton emit 16-byte vectorized fp32 loads and 4-byte int8 stores, with the per-row absmax reduced along axis 1.: 8.07x (1.376 ms vs 11.108 ms)
- #2 [speed] Streaming cache hints (evict_first on the load and the int8 and scale stores) and larger tiles of 2048 elements with 8 warps, to squeeze the last bit of DRAM bandwidth on this already-bandwidth-bound kernel.: 8.07x (1.377 ms vs 11.103 ms)

## known
- Strategy for this step: Make global-memory access efficient: coalesced, aligned, vectorized loads and stores; avoid redundant reads.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
