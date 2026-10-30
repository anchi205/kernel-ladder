import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"ROWS": 1}, num_warps=4),
        triton.Config({"ROWS": 1}, num_warps=8),
        triton.Config({"ROWS": 1}, num_warps=16),
        triton.Config({"ROWS": 2}, num_warps=4),
        triton.Config({"ROWS": 2}, num_warps=8),
        triton.Config({"ROWS": 2}, num_warps=16),
        triton.Config({"ROWS": 4}, num_warps=8),
        triton.Config({"ROWS": 4}, num_warps=16),
    ],
    key=["N"],
)
@triton.jit
def _rmsnorm_kernel(x_ptr, w_ptr, out_ptr, n_rows, N, eps,
                    BLOCK: tl.constexpr, ROWS: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    cmask = cols < N
    w = tl.load(w_ptr + cols, mask=cmask, other=0.0,
                eviction_policy="evict_last").to(tl.float32)
    for r in tl.static_range(ROWS):
        row = pid * ROWS + r
        mask = cmask & (row < n_rows)
        base = row * N
        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)
        ms = tl.sum(x * x, axis=0) / N
        inv = 1.0 / tl.sqrt(ms + eps)
        y = x * inv * w
        tl.store(out_ptr + base + cols, y.to(out_ptr.dtype.element_ty),
                 mask=mask, cache_modifier=".cs")


def forward(x, w, eps: float = 1e-6):
    orig_shape = x.shape
    N = orig_shape[-1]
    x2 = x.contiguous().view(-1, N)
    w = w.contiguous()
    out = torch.empty_like(x2)
    rows = x2.shape[0]
    BLOCK = triton.next_power_of_2(N)
    grid = lambda meta: (triton.cdiv(rows, meta["ROWS"]),)
    _rmsnorm_kernel[grid](x2, w, out, rows, N, eps, BLOCK=BLOCK)
    return out.view(orig_shape)
