# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 0.00x faster
best      attempt #1, 3.46x (Straightforward Triton row-wise RMSNorm, one program per row, whole row loaded in a single block (padded to the next power of two), fp32 accumulation.)

## tried
- #1 [ok] Straightforward Triton row-wise RMSNorm, one program per row, whole row loaded in a single block (padded to the next power of two), fp32 accumulation.: green

## known
- Write a straightforward, correct first version of this kernel. Do NOT optimize yet: other steps will apply specific optimizations to your code afterwards.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
