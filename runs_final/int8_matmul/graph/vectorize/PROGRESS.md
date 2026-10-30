# PROGRESS

goal      candidate for 'int8_matmul' that matches the reference and runs >= 5.56x faster
best      attempt #2, 6.73x (Pre-transpose B so both int8 operands are K-contiguous, which suits the int8 MMA's operand layout and avoids a slow MN-major shared-memory path for B. The kernel then uses `tl.trans` on the B tile, and I added a few extra tile configs.)

## tried
- #1 [speed] Larger autotuned tiles (128x128 and up, BLOCK_K 64–128) with grouped L2 swizzle, pointer-increment loads with EVEN_K unmasked K loads, and int32 accumulation, so int8 global loads are contiguous and coalesced and tiles are reused across blocks.: 4.52x (0.688 ms vs 3.111 ms)
- #2 [ok] Pre-transpose B so both int8 operands are K-contiguous, which suits the int8 MMA's operand layout and avoids a slow MN-major shared-memory path for B. The kernel then uses `tl.trans` on the B tile, and I added a few extra tile configs.: green

## known
- Strategy for this step: Make global-memory access efficient: coalesced, aligned, vectorized loads and stores; avoid redundant reads.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
