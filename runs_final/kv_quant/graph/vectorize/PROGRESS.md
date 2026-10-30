# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 8.58x faster
best      attempt #1, 8.08x (Process several rows per program as a 2D [ROWS, BLOCK] tile. Each thread gets 16B vector loads and 4B int8 stores, with more bytes in flight per CTA and fewer programs.)

## tried
- #1 [speed] Process several rows per program as a 2D [ROWS, BLOCK] tile. Each thread gets 16B vector loads and 4B int8 stores, with more bytes in flight per CTA and fewer programs.: 8.08x (1.375 ms vs 11.115 ms)
- #2 [speed] Use larger row tiles (16 rows × 128, about 2048 elements per CTA) so each thread has more 16B loads in flight, and add evict_first cache hints on the streaming load and stores to cut cache pollution.: 5.42x (2.048 ms vs 11.104 ms)

## known
- Strategy for this step: Make global-memory access efficient: coalesced, aligned, vectorized loads and stores; avoid redundant reads.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
