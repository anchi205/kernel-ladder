# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.53x faster
best      attempt #1, 3.47x (Add Triton alignment hints (`multiple_of`, `max_contiguous`) plus `evict_first` on the streamed x loads and output stores, and `evict_last` on the reused weight, to get 128-bit vectorized, cache-friendly global accesses.)

## tried
- #1 [speed] Add Triton alignment hints (`multiple_of`, `max_contiguous`) plus `evict_first` on the streamed x loads and output stores, and `evict_last` on the reused weight, to get 128-bit vectorized, cache-friendly global accesses.: 3.47x (1.123 ms vs 3.896 ms)
- #2 [speed] Process 2 rows per program with a 2D tile, so each thread has twice as many independent 128-bit loads in flight. This aims to close the last few percent to HBM peak. Hints are dropped where Triton already infers them, and the evict policies are kept.: 3.40x (1.144 ms vs 3.889 ms)

## known
- Strategy for this step: Make global-memory access efficient: coalesced, aligned, vectorized loads and stores; avoid redundant reads.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
