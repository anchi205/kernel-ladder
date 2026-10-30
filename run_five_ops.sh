#!/usr/bin/env bash
# Run the strategy graph for each op, one at a time, into runs_final/.
cd "$(dirname "$0")"
: "${KL_SSH_TARGET:?set KL_SSH_TARGET (user@host)}" "${KL_CONTAINER:?set KL_CONTAINER}"   # see REMOTE_SETUP.md
export KL_SSH_TARGET KL_CONTAINER KL_SSH_KEY="${KL_SSH_KEY:-$HOME/.ssh/id_ed25519}"
mkdir -p runs_final
export KL_GPU_BUDGET_MIN=8 PYTHONUNBUFFERED=1
for op in softmax rmsnorm kv_quant int8_matmul attention; do
  echo "=== $op: $(date +%H:%M:%S)" | tee -a runs_final/progress.log
  .venv/bin/kernelladder graph "$op" --remote --max-attempts 2 --workers 3 --out runs_final > "runs_final/$op.log" 2>&1
  echo "=== $op finished (exit $?) $(date +%H:%M:%S)" | tee -a runs_final/progress.log
done
echo "=== all done" | tee -a runs_final/progress.log
