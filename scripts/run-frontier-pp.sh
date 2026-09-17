#!/usr/bin/env bash
# Frontier MI250X: FFTMatvec hipcc vs SCALE-amd Pennycook benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=frontier-workdirs.sh
source "${SCRIPT_DIR}/frontier-workdirs.sh"

cd "${FFT_MVEC_SRC}"

HIP_BIN="${BUILD_HIP}/fft_matvec"
SCALE_BIN="${BUILD_SCALE}/fft_matvec"

AMD_VER="${AMD_VER:-7.2.0}"
ROCM_VER="${ROCM_VER:-7.2.0}"
LIBFABRIC_VER="${LIBFABRIC_VER:-}"

ml unload darshan-runtime 2>/dev/null || true
if [[ -n "$LIBFABRIC_VER" ]]; then
  ml PrgEnv-amd cray-hdf5-parallel "amd/${AMD_VER}" "rocm/${ROCM_VER}" "libfabric/${LIBFABRIC_VER}" xpmem
else
  ml PrgEnv-amd cray-hdf5-parallel "amd/${AMD_VER}" "rocm/${ROCM_VER}" xpmem
fi
export LIBRARY_PATH=/opt/xpmem/lib64:${LIBRARY_PATH:-}

SCRATCH="${SCRATCH:-${FFT_MVEC_WORK}}"
SCALE_ROOT="${SCALE_ROOT:-$HOME/scale-1.6.1-Linux}"
LOG_DIR="$SCRATCH/logs"
mkdir -p "$LOG_DIR"

export ROCR_VISIBLE_DEVICES=0
MODULE_LD_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ":" "\n" | grep -v scale | paste -sd: -)

# HIP builds use RCCL. SCALE CUDA builds host-stage MPI (Cray GTL rejects
# SCALE cudaMalloc pointers). Set MPICH_GPU_SUPPORT_ENABLED=1 only for native
# HIP device-MPI experiments; it is not required for FFTMatvec multiproc.

NM="${NM:-10000}"
ND=100
NT="${NT:-1000}"

run_one() {
  local label="$1"
  local bin="$2"
  local log="$LOG_DIR/${label}_nm${NM}.log"
  echo "=== $label nm=$NM ===" | tee "$log"
  if [[ "$label" == "scale-amd" ]]; then
    export LD_LIBRARY_PATH="${SCALE_ROOT}/targets/gfx90a/lib:${MODULE_LD_PATH}"
  else
    export LD_LIBRARY_PATH="${MODULE_LD_PATH}"
  fi
  if ! srun -n 1 -u "$bin" -nm "$NM" -nd "$ND" -Nt "$NT" -t -raw 2>&1 | tee -a "$log"; then
    echo "FAILED: $label nm=$NM" | tee -a "$log"
    return 1
  fi
  grep -E "test passed|Matvec test" "$log" | tee -a "$log" || true
}

passes() {
  local log="$1"
  grep -q "F Matvec test passed" "$log" && grep -q "F\* Matvec test passed" "$log"
}

if [[ "${SKIP_SWEEP:-0}" != "1" ]]; then
  NM=""
  for try in 10000 5000 3000 2000 1000 500; do
    hip_log=$(mktemp)
    scale_log=$(mktemp)
    export LD_LIBRARY_PATH="${MODULE_LD_PATH}"
    if srun -n 1 -u "${HIP_BIN}" -nm "$try" -nd "$ND" -Nt "$NT" -t -raw >"$hip_log" 2>&1 && passes "$hip_log"; then
      export LD_LIBRARY_PATH="${SCALE_ROOT}/targets/gfx90a/lib:${MODULE_LD_PATH}"
      if srun -n 1 -u "${SCALE_BIN}" -nm "$try" -nd "$ND" -Nt "$NT" -t -raw >"$scale_log" 2>&1 && passes "$scale_log"; then
        NM="$try"
        rm -f "$hip_log" "$scale_log"
        break
      fi
    fi
    rm -f "$hip_log" "$scale_log"
  done
  if [[ -z "$NM" ]]; then
    echo "ERROR: no common nm found for hipcc and scale-amd" >&2
    exit 1
  fi
  echo "Using nm=$NM Nt=$NT"
fi

run_one hipcc "${HIP_BIN}"
run_one scale-amd "${SCALE_BIN}"

python3 "${FFT_MVEC_SRC}/scripts/compute_pennycook_fftmatvec.py" \
  "$LOG_DIR/hipcc_nm${NM}.log" \
  "$LOG_DIR/scale-amd_nm${NM}.log" \
  "$SCRATCH/pennycook_hipcc_vs_scale-amd_nm${NM}_raw.csv" \
  "${PP_METRIC:-pipeline}"
