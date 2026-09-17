#!/usr/bin/env bash
# Capture RedSCALE profiles for FFTMatvec on Frontier (MI250X / gfx90a).
# Requires an active allocation (salloc/sbatch). Uses nm=5000 by default to
# match the prior hipcc vs SCALE-amd phase breakdown.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=frontier-workdirs.sh
source "${SCRIPT_DIR}/frontier-workdirs.sh"

cd "${FFT_MVEC_SRC}"

SCALE_BIN="${BUILD_SCALE}/fft_matvec"
SCALE_ROOT="${SCALE_ROOT:-$HOME/scale-1.6.1-Linux}"
PROFILER="${SCALE_ROOT}/bin/redscale_profiler"

AMD_VER="${AMD_VER:-7.2.0}"
ROCM_VER="${ROCM_VER:-7.2.0}"

ml unload darshan-runtime 2>/dev/null || true
ml PrgEnv-amd cray-hdf5-parallel "amd/${AMD_VER}" "rocm/${ROCM_VER}" xpmem
export LIBRARY_PATH=/opt/xpmem/lib64:${LIBRARY_PATH:-}

export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
MODULE_LD_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ":" "\n" | grep -v scale | paste -sd: -)
export LD_LIBRARY_PATH="${SCALE_ROOT}/targets/gfx90a/lib:${MODULE_LD_PATH}"
# SCALE multiproc uses host-staged MPI (see gpu_collectives_mpi_cuda.cpp).

NM="${NM:-5000}"
ND="${ND:-100}"
NT="${NT:-1000}"
# Fewer matvecs under the profiler keeps profiles smaller; override with NMATVEC.
NMATVEC="${NMATVEC:-20}"

PROFILE_DIR="${PROFILE_DIR:-${FFT_MVEC_WORK}/profiles}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${PROFILE_DIR}/${STAMP}"
mkdir -p "$OUT_DIR"

if [[ ! -x "$SCALE_BIN" ]]; then
  echo "ERROR: missing SCALE binary: $SCALE_BIN" >&2
  exit 1
fi
if [[ ! -x "$PROFILER" ]]; then
  echo "ERROR: missing redscale_profiler: $PROFILER" >&2
  exit 1
fi
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: no SLURM allocation; wait for salloc or sbatch first." >&2
  exit 1
fi

# Doc uses REDSCALE_PROFILE; redscale_profiler --help also mentions SCALE_PROFILE.
# Set both. Queue modes: device (serialize), queue (per-stream), none (API-only).
QUEUE_MODES="${QUEUE_MODES:-queue device}"

run_profile() {
  local mode="$1"
  local tag="nm${NM}_nmatvec${NMATVEC}_q${mode}"
  local prof="${OUT_DIR}/scale_${tag}.prof"
  local log="${OUT_DIR}/scale_${tag}.log"

  echo "=== capturing ${tag} -> ${prof} ===" | tee "$log"
  export REDSCALE_PROFILE="$prof"
  export SCALE_PROFILE="$prof"
  export REDSCALE_PROFILE_KERNELS="${REDSCALE_PROFILE_KERNELS:-1}"
  export REDSCALE_PROFILE_MEMOPS="${REDSCALE_PROFILE_MEMOPS:-1}"
  export REDSCALE_PROFILE_QUEUE_MODE="$mode"

  # Attach to the existing salloc/sbatch. fft_matvec -N is --reps (timed matvecs).
  local srun_base=(srun --jobid="${SLURM_JOB_ID}" --export=ALL -n 1 -c 7 --gpus-per-task=1 --gpu-bind=closest -u)
  set +e
  "${srun_base[@]}" "$SCALE_BIN" -nm "$NM" -nd "$ND" -Nt "$NT" -N "$NMATVEC" -t -raw \
    >"$log" 2>&1
  local rc=$?
  set -e

  unset REDSCALE_PROFILE SCALE_PROFILE REDSCALE_PROFILE_QUEUE_MODE

  if [[ $rc -ne 0 ]]; then
    echo "FAILED profile capture (${tag}), rc=$rc" | tee -a "$log"
    return "$rc"
  fi
  if [[ ! -s "$prof" ]]; then
    echo "ERROR: empty/missing profile $prof" | tee -a "$log"
    return 1
  fi
  ls -lh "$prof" | tee -a "$log"
}

summarize() {
  local prof="$1"
  local base="${prof%.prof}"
  echo "=== summarizing $(basename "$prof") ==="
  # Flags must precede <profile>.
  "$PROFILER" --no-colour summary -C -H "$prof" \
    >"${base}_kernels.txt" 2>&1 || true
  "$PROFILER" --no-colour summary -a -H "$prof" \
    >"${base}_api.txt" 2>&1 || true
  "$PROFILER" --no-colour summary -H -m "$prof" \
    >"${base}_memops.txt" 2>&1 || true
  "$PROFILER" --no-colour summary -a -m -C -H "$prof" \
    >"${base}_full.txt" 2>&1 || true
}

echo "host=$(hostname) job=${SLURM_JOB_ID} out=${OUT_DIR}"
echo "bin=${SCALE_BIN}"
echo "nm=${NM} nd=${ND} Nt=${NT} Nmatvec=${NMATVEC}"
echo "queue_modes=${QUEUE_MODES}"

for mode in $QUEUE_MODES; do
  run_profile "$mode"
done

for prof in "${OUT_DIR}"/scale_*.prof; do
  [[ -e "$prof" ]] || continue
  summarize "$prof"
done

# Quick pointers for the slowdown investigation (FFT / Geam / SBGEMV).
INDEX="${OUT_DIR}/INDEX.txt"
{
  echo "SCALE RedSCALE profiles: ${OUT_DIR}"
  echo "Problem: nm=${NM} nd=${ND} Nt=${NT} Nmatvec=${NMATVEC}"
  echo
  echo "Files:"
  ls -1 "$OUT_DIR"
  echo
  echo "Look first at *_api.txt for cuFFT/cublas/cudaMemcpy overhead,"
  echo "and *_kernels.txt for which kernels ran (hipBLAS vs SCALE-local)."
} >"$INDEX"

echo "DONE. Index: $INDEX"
echo "$OUT_DIR" >"${PROFILE_DIR}/latest"
cat "$INDEX"
