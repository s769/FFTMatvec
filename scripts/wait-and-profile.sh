#!/usr/bin/env bash
# Wait for salloc job to start, then capture + summarize SCALE RedSCALE profiles.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=frontier-workdirs.sh
source "${SCRIPT_DIR}/frontier-workdirs.sh"

JOBID="${JOBID:-5492679}"
POLL_SEC="${POLL_SEC:-30}"
LOG_DIR="${FFT_MVEC_WORK}/logs"
mkdir -p "$LOG_DIR"
WAIT_LOG="${LOG_DIR}/wait_profile_${JOBID}.log"

echo "Waiting for job ${JOBID} ..." | tee "$WAIT_LOG"

while true; do
  state=$(squeue -j "$JOBID" -h -o '%T' 2>/dev/null || true)
  reason=$(squeue -j "$JOBID" -h -o '%R' 2>/dev/null || true)
  if [[ -z "$state" ]]; then
    echo "$(date -Is) job ${JOBID} no longer in queue" | tee -a "$WAIT_LOG"
    exit 1
  fi
  echo "$(date -Is) state=${state} reason=${reason}" | tee -a "$WAIT_LOG"
  case "$state" in
    RUNNING) break ;;
    COMPLETING|COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|PREEMPTED)
      echo "job ended in state=${state}" | tee -a "$WAIT_LOG"
      exit 1
      ;;
  esac
  sleep "$POLL_SEC"
done

nodelist=$(squeue -j "$JOBID" -h -o '%N')
echo "Allocation RUNNING on ${nodelist}; starting profile capture" | tee -a "$WAIT_LOG"

export SLURM_JOB_ID="$JOBID"
# Prefer the ROCm-7.2 HIP tree used in the final report when present.
export BUILD_HIP="${BUILD_HIP:-${FFT_MVEC_WORK}/build_hip72}"

bash "${SCRIPT_DIR}/profile-frontier-scale.sh" 2>&1 | tee -a "$WAIT_LOG"
