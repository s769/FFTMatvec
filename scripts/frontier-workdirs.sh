# Source from FFTMatvec scripts: keeps builds, logs, and caches off $HOME.
# Usage: source "$(dirname "$0")/frontier-workdirs.sh"

FFT_MVEC_SRC="${FFT_MVEC_SRC:-$HOME/FFTMatvec}"

# Prefer explicit FFT_MVEC_WORK. Else $MEMBERWORK/FFTMatvec (OLCF sets MEMBERWORK),
# else per-user scratch.
if [[ -z "${FFT_MVEC_WORK:-}" ]]; then
  if [[ -n "${MEMBERWORK:-}" ]]; then
    FFT_MVEC_WORK="${MEMBERWORK}/FFTMatvec"
  else
    FFT_MVEC_WORK="/lustre/orion/scratch/${USER}/FFTMatvec"
  fi
fi

BUILD_HIP="${BUILD_HIP:-${FFT_MVEC_WORK}/build_hip}"
BUILD_SCALE="${BUILD_SCALE:-${FFT_MVEC_WORK}/build}"
SCRATCH="${SCRATCH:-${FFT_MVEC_WORK}}"

mkdir -p "${FFT_MVEC_WORK}/logs" "${BUILD_HIP}" "${BUILD_SCALE}"

# Avoid writing multi-GB core files to $HOME on crashes
ulimit -c 0 2>/dev/null || true
