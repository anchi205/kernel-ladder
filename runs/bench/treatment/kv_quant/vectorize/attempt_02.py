import torch
import triton
import triton.language as tl


@triton.jit
def _kv_quant_kernel(x_ptr, q_ptr, s_ptr, n_rows, D, n_tiles, nprog,
                     ROWS: tl.constexpr, BLOCK_D: tl.constexpr,
                     EVEN: tl.constexpr, STAGES: tl.constexpr):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    cols = tl.max_contiguous(tl.multiple_of(cols, 4), BLOCK_D)
    r_idx = tl.arange(0, ROWS)

    for t in tl.range(pid, n_tiles, nprog, num_stages=STAGES):
        rows = t * ROWS + r_idx
        rmask = rows < n_rows
        offs = rows[:, None].to(tl.int64) * D + cols[None, :]

        if EVEN:
            mask = rmask[:, None] & (cols[None, :] < BLOCK_D)
        else:
            mask = rmask[:, None] & (cols[None, :] < D)

        x = tl.load(x_ptr + offs, mask=mask, other=0.0,
                    eviction_policy="evict_first").to(tl.float32)
        amax = tl.max(tl.abs(x), axis=1)
        amax = tl.maximum(amax, 1e-8)
        scale = tl.math.div_rn(amax, 127.0)

        y = tl.math.div_rn(x, scale[:, None])
        # round half to even (matches torch.round)
        r = tl.floor(y + 0.5)
        tie = (r - y) == 0.5
        odd = (r - 2.0 * tl.floor(r * 0.5)) == 1.0
        r = tl.where(tie & odd, r - 1.0, r)
        r = tl.minimum(tl.maximum(r, -127.0), 127.0)

        tl.store(q_ptr + offs, r.to(tl.int8), mask=mask,
                 eviction_policy="evict_first")
        tl.store(s_ptr + rows, scale, mask=rmask)


_SM_COUNT = {}


def forward(x):
    x = x.contiguous()
    if x.dtype != torch.float32:
        x = x.float()
    D = x.shape[-1]
    n_rows = x.numel() // D
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    s = torch.empty(x.shape[:-1], dtype=torch.float32, device=x.device)
    BLOCK_D = triton.next_power_of_2(D)
    ROWS = 8
    n_tiles = triton.cdiv(n_rows, ROWS)
    dev = x.device.index if x.device.index is not None else torch.cuda.current_device()
    if dev not in _SM_COUNT:
        _SM_COUNT[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    nprog = max(1, min(n_tiles, _SM_COUNT[dev] * 8))
    _kv_quant_kernel[(nprog,)](x, q, s, n_rows, D, n_tiles, nprog,
                               ROWS=ROWS, BLOCK_D=BLOCK_D,
                               EVEN=(BLOCK_D == D), STAGES=3, num_warps=4)
    return q, s
