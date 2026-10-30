# PROGRESS

goal      candidate for 'attention' that matches the reference and runs >= 0.00x faster
best      attempt #2, 6.53x (Cast the softmax probabilities `p` to V's dtype before the PV dot. The last attempt failed to compile on fp16 inputs because it fed fp32 `p` against fp16 `v`. The cast also mirrors the reference's `.to(q.dtype)`.)

## tried
- #1 [correctness] Straightforward causal flash-attention Triton kernel with online softmax, fp32 accumulation, and IEEE-precision dots for matching the reference.: CompilationError('def _attn_fwd(\n    Q, K, V, O,\n    N, sm_scale,\n    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,\n):\n    pid_m = tl.prog
- #2 [ok] Cast the softmax probabilities `p` to V's dtype before the PV dot. The last attempt failed to compile on fp16 inputs because it fed fp32 `p` against fp16 `v`. The cast also mirrors the reference's `.to(q.dtype)`.: green

## known
- Write a straightforward, correct first version of this kernel. Do NOT optimize yet: other steps will apply specific optimizations to your code afterwards.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
