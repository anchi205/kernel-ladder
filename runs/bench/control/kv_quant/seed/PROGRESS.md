# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 0.00x faster
best      attempt #1, 8.30x (Straightforward Triton kernel, one program per row. It computes the absmax, then the scale with IEEE division, then round-half-even via libdevice rint, clamps, and stores int8 plus the per-row scale.)

## tried
- #1 [ok] Straightforward Triton kernel, one program per row. It computes the absmax, then the scale with IEEE division, then round-half-even via libdevice rint, clamps, and stores int8 plus the per-row scale.: green

## known
- Write a straightforward, correct first version of this kernel. Do NOT optimize yet: other steps will apply specific optimizations to your code afterwards.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
