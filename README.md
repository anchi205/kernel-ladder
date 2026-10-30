# KernelLadder

An agent that writes and optimizes GPU kernels and proves every speedup, built on the
harness / loop / graph ladder. This stage is the check: correctness against a fixed
reference, then timing against it, with EXPECTED / SAW / FENCE / DEAD ENDS feedback.

## Ops

| op | what it is | reference |
|---|---|---|
| `softmax` | row softmax | torch's fused kernel (hard to beat) |
| `rmsnorm` | RMS normalization | unfused eager ops (fusion wins) |
| `attention` | causal scaled dot-product attention | naive: materializes the score matrix |
| `int8_matmul` | int8 x int8 with per-row/col scales | float matmul + 2 scale multiplies |
| `kv_quant` | quantize new K/V rows to int8 + scales (2 outputs) | 8 eager kernels |

## Setup

    uv venv && uv pip install -e ".[dev]"
    pytest -q          # CPU-side tests; GPU tests skip without KL_SSH_TARGET

The remote GPU box needs its own environment: see REMOTE_SETUP.md.

## Run checks on a remote GPU container

    export KL_SSH_TARGET=user@host KL_CONTAINER=<container> KL_SSH_KEY=~/.ssh/id_ed25519
    # optional: KL_REMOTE_PYTHON (default ~/kl-venv/bin/python), KL_REMOTE_DIR (default ~/work/kernelladder)
    kernelladder check softmax candidates/softmax_stable.py --remote --min-speedup 0

The verifier and reference ops are synced to the box on each run; the candidate is sent on stdin.
Remote tests skip unless `KL_SSH_TARGET` is set.

## Let Claude write the kernel

    kernelladder optimize rmsnorm --remote --target 1.5 --max-attempts 8      # --proposer code (default)

The default proposer runs the local `claude -p` CLI (Claude Code, logged in with a subscription), with tools off in an
empty directory. All results in RESULTS.md came from it. `--proposer api` uses the Anthropic API instead and needs
`ANTHROPIC_API_KEY` (or `ant auth login`). The agent is nondeterministic: re-running gives different kernels, while
the checker itself is deterministic.

## The parallel strategy graph

    kernelladder graph attention --remote --max-attempts 2

seed -> strategy nodes (parallel where they commute) -> combine -> merge (plain code) -> gate.
Model calls run in parallel; GPU checks hold one lock. `compare` times a result against torch.compile and
PyTorch's own fused ops. `gate` classifies a result by undo cost: ships on green / needs a wide sweep plus an
independent reviewer / needs a human.

## Learning across runs

    kernelladder graph kv_quant --remote --lessons lessons.json --learn   # apply and add lessons
    kernelladder lessons lessons.json                                     # show them
    kernelladder lessons lessons.json --drop <run-id>                     # revert one run's lessons
    kernelladder bench --train rmsnorm --heldout kv_quant --remote        # held-out evaluation

Lessons are derived by code from structured run data, filed per GPU architecture and op family, and only change
the search (attempts per strategy, hints). They never touch the verifier or the gates.

## GPU guard

Every remote run checks that the GPU is idle first, alerts on your Mac when it is busy or when usage is high, and is
logged. `kernelladder usage` shows total GPU time held. Env: KL_BUSY_UTIL, KL_GPU_BUDGET_MIN (per process),
KL_GPU_DAILY_MIN (across sessions), KL_USAGE_LOG.

See RESULTS.md for measured results against PyTorch baselines and their limits.
