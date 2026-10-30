# PROGRESS

goal      candidate for 'softmax' that matches the reference and runs >= 0.98x faster
best      attempt #1, 0.96x (Single-pass row softmax with the whole row held in registers. A constexpr EVEN_N flag drops the masks when the row length is a power of two, so loads and stores can vectorize to 128-bit. Loads use evict_first and stores use a streaming cache hint. exp2 with a folded log2e constant replaces exp, and a reciprocal multiply replaces the per-element divide. Warps are set so more rows are in flight per SM.)

## tried
- #1 [speed] Single-pass row softmax with the whole row held in registers. A constexpr EVEN_N flag drops the masks when the row length is a power of two, so loads and stores can vectorize to 128-bit. Loads use evict_first and stores use a streaming cache hint. exp2 with a folded log2e constant replaces exp, and a reciprocal multiply replaces the per-element divide. Warps are set so more rows are in flight per SM.: 0.96x (1.141 ms vs 1.099 ms)
- #2 [speed] Row softmax with plain loads and stores (no eviction or streaming hints, which can hurt on GB10's unified LPDDR memory), 8 warps per row to cut registers per thread and raise resident warps, and rows processed in a grid-stride loop so each program handles several rows.: 0.95x (1.139 ms vs 1.077 ms)

## known
- Strategy for this step: Make global-memory access efficient: coalesced, aligned, vectorized loads and stores; avoid redundant reads.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
