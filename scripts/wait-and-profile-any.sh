#!/usr/bin/env bash
# Wait for any of the given SLURM job IDs to become RUNNING, then profile.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=frontier-workdirs.sh
source "${SCRIPT_DIR}/frontier-workdirs.sh"

JOBIDS=("$@")
if [[ ${#JOBIDS[@]} -eq 0 ]]; then
  echo "usage: $0 <jobid> [jobid...]" >&2
  exit 2
fi

POLL_SEC="${POLL_SEC:-15}"
LOG_DIR="${FFT_MVEC_WORK}/logs"
mkdir -p "$LOG_DIR"
WAIT_LOG="${LOG_DIR}/wait_profile_any_$(date +%Y%m%d_%H%M%S).log"

echo "Watching jobs: ${JOBIDS[*]}" | tee "$WAIT_LOG"

pick_running() {
  local j state
  for j in "${JOBIDS[@]}"; do
    state=$(squeue -j "$j" -h -o '%T' 2>/dev/null || true)
    if [[ "$state" == "RUNNING" ]]; then
      echo "$j"
      return 0
    fi
  done
  return 1
}

all_gone() {
  local j state any=0
  for j in "${JOBIDS[@]}"; do
    state=$(squeue -j "$j" -h -o '%T' 2>/dev/null || true)
    if [[ -n "$state" ]]; then
      any=1
      case "$state" in
        COMPLETING|COMPLETED|CANCELLED|FAILED|TIMEOUT|NODE_FAIL|PREEMPTED) ;;
        *) return 1 ;;
      esac
    fi
  done
  [[ $any -eq 0 ]]
}

while true; do
  line=$(date -Is)
  for j in "${JOBIDS[@]}"; do
    st=$(squeue -j "$j" -h -o '%T %R' 2>/dev/null || echo 'GONE')
    line+=" | ${j}:${st}"
  done
  echo "$line" | tee -a "$WAIT_LOG"

  if jobid=$(pick_running); then
    nodelist=$(squeue -j "$jobid" -h -o '%N')
    echo "Using job ${jobid} on ${nodelist}" | tee -a "$WAIT_LOG"
    export SLURM_JOB_ID="$jobid"
    export BUILD_HIP="${BUILD_HIP:-${FFT_MVEC_WORK}/build_hip72}"
    bash "${SCRIPT_DIR}/profile-frontier-scale.sh" 2>&1 | tee -a "$WAIT_LOG"
    exit $?
  fi

  if all_gone; then
    echo "All watched jobs gone without running" | tee -a "$WAIT_LOG"
    exit 1
  fi
  sleep "$POLL_SEC"
done
