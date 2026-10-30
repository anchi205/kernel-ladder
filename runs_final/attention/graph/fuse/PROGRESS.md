# PROGRESS

goal      candidate for 'attention' that matches the reference and runs >= 6.66x faster
best      attempt #2, 7.14x (Keep the fused flash-attention kernel and fix the compile error by casting `p` to V's dtype before the PV dot, since fp16 inputs hit a mixed fp32/fp16 dot. The cast is a no-op for fp32.)

## tried
- #1 [correctness] Keep the single fused flash-attention kernel and cut its overhead: 64x32 tiles, exp2 with the scale folded in, an unmasked fast loop over fully visible K/V blocks plus a masked loop only on the diagonal, and heavy query blocks scheduled first.: CompilationError('def _attn_fwd(\n    Q, K, V, O,\n    N, qk_scale,\n    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr,\n):\n    num_m = tl.cdiv
- #2 [ok] Keep the fused flash-attention kernel and fix the compile error by casting `p` to V's dtype before the PV dot, since fp16 inputs hit a mixed fp32/fp16 dot. The cast is a no-op for fp32.: green

## known
- Strategy for this step: Fuse separate kernels and intermediate tensors into as few kernels as possible, so intermediate results never round-trip through global memory.

## never
- calling the op's own library function
- importing kernelladder
- editing anything but the candidate file
