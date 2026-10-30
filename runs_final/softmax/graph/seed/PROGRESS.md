# PROGRESS

goal      candidate for 'softmax' that matches the reference and runs >= 0.00x faster
best      attempt #1, 0.96x (Straightforward row-wise Triton softmax: one program per row, loads the whole row (padded to next power of 2), subtracts the row max, exponentiates, and normalizes by the sum.)

## tried
- #1 [ok] Straightforward row-wise Triton softmax: one program per row, loads the whole row (padded to next power of 2), subtracts the row max, exponentiates, and normalizes by the sum.: green

## known
- Write a straightforward, correct first version of this kernel. Do NOT optimize yet: other steps will apply specific optimizations to your code afterwards.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
