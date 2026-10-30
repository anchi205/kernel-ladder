# PROGRESS

goal      candidate for 'kv_quant' that matches the reference and runs >= 0.00x faster
best      attempt #1, 8.22x (Fused single-pass Triton kernel that handles several rows per program, computing the absmax, scale, round-half-even and int8 cast in registers. It uses IEEE-rounded division to match the reference.)

## tried
- #1 [ok] Fused single-pass Triton kernel that handles several rows per program, computing the absmax, scale, round-half-even and int8 cast in registers. It uses IEEE-rounded division to match the reference.: green

## known
- Write a straightforward, correct first version of this kernel. Do NOT optimize yet: other steps will apply specific optimizations to your code afterwards.
- Lesson (sm_121, memory-bound): 'fuse' gained on 1 of 1 op(s), up to 1.07x.
- Lesson (sm_121, memory-bound): 'vectorize' gained on 1 of 1 op(s), up to 1.05x.
- Lesson (sm_121, memory-bound): 'reduce' gained on 1 of 1 op(s), up to 1.04x.
- Lesson (sm_121, memory-bound): 'launch' gained nothing on 1 op(s) tried.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
