# Remote GPU setup

KernelLadder runs candidates on a GPU machine reached over SSH, inside a Docker container:
`ssh <target> docker exec -i <container> <python> ...`. The verifier and reference ops are copied there on every run;
only the candidate file is sent on stdin.

## What the box needs
* Linux, NVIDIA GPU, a running container you can `docker exec` into, SSH key access to the host.
* A Python 3.12 venv with a CUDA build of torch and Triton. Versions used for RESULTS.md:
  Python 3.12.12, torch 2.14.1+cu130, triton 3.8.0, CUDA 13.0, one GB10 (sm_121).
* Python headers for the interpreter (Triton compiles a small C helper). The container's system Python 3.13 had none, so
  the venv uses a uv-managed Python 3.12.

## Setup used (inside the container)

    uv venv --python 3.12 ~/kl-venv
    uv pip install --python ~/kl-venv/bin/python torch numpy pytest triton   # a CUDA build of torch

## Point KernelLadder at it (on your machine)

    export KL_SSH_TARGET=user@host KL_CONTAINER=<container> KL_SSH_KEY=~/.ssh/id_ed25519
    # optional: KL_REMOTE_PYTHON (default ~/kl-venv/bin/python), KL_REMOTE_DIR (default ~/work/kernelladder)
    kernelladder check softmax candidates/softmax_stable.py --remote --min-speedup 0
    pytest -q          # the remote tests now run too

## Limits of the box used
* `ncu` and `compute-sanitizer` do not support sm_121, so the race check is a determinism test (identical output on
  repeated runs), not a race detector.
* Runs are capped by `KL_MAX_RAM_GB` (default 16) and `KL_MAX_GPU_FRAC` (default 0.25), killed after a timeout, and
  guarded by `KL_BUSY_UTIL`, `KL_GPU_BUDGET_MIN`, `KL_GPU_DAILY_MIN`, `KL_USAGE_LOG`.
