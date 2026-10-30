# PROGRESS

goal      candidate for 'int8_matmul' that matches the reference and runs >= 0.00x faster
best      attempt #1, 5.46x (Plain Triton tiled matmul on int8 tiles with an int32 accumulator, then a fused per-row and per-column fp32 scale in the epilogue (first correct version, no tuning).)

## tried
- #1 [ok] Plain Triton tiled matmul on int8 tiles with an int32 accumulator, then a fused per-row and per-column fp32 scale in the epilogue (first correct version, no tuning).: green

## known
- Write a straightforward, correct first version of this kernel. Do NOT optimize yet: other steps will apply specific optimizations to your code afterwards.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
