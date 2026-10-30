# PROGRESS

goal      candidate for 'rmsnorm' that matches the reference and runs >= 0.00x faster
best      attempt #1, 3.24x (Straightforward row-wise Triton RMSNorm: one program per row, loop over column blocks to accumulate sum of squares in fp32, then a second pass to normalize and scale by the weight.)

## tried
- #1 [ok] Straightforward row-wise Triton RMSNorm: one program per row, loop over column blocks to accumulate sum of squares in fp32, then a second pass to normalize and scale by the weight.: green

## known
- Write a straightforward, correct first version of this kernel. Do NOT optimize yet: other steps will apply specific optimizations to your code afterwards.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
