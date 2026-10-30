# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 3.54x faster
best      attempt #1, 3.47x (Merge the best single-pass row kernel (nw=8 at 4096) with the vectorize variant's eviction hints (`evict_first` for x and out, `evict_last` for w), `tl.rsqrt`, and int64 row offsets, and keep the looped fallback for rows wider than 8192.)

## tried
- #1 [speed] Merge the best single-pass row kernel (nw=8 at 4096) with the vectorize variant's eviction hints (`evict_first` for x and out, `evict_last` for w), `tl.rsqrt`, and int64 row offsets, and keep the looped fallback for rows wider than 8192.: 3.47x (1.134 ms vs 3.937 ms)
- #2 [speed] The row kernel is already at about 99% of peak bandwidth (243 of 246 GB/s), so I'm trimming the host-side overhead that accounts for the 1.134 ms measured vs 1.104 ms kernel gap. This skips redundant contiguous/view calls and does the next-power-of-2 and warp-count dispatch with plain integer ops. The kernel body is unchanged.: 3.43x (1.127 ms vs 3.864 ms)

## known
- Combine the optimizations from several variants.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
