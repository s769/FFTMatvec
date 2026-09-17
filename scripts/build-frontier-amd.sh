#!/usr/bin/env bash
# Rebuild FFTMatvec hipcc + SCALE-amd on Frontier (mpi-scale branch)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=frontier-workdirs.sh
source "${SCRIPT_DIR}/frontier-workdirs.sh"

cd "${FFT_MVEC_SRC}"

SCALE_ROOT="${SCALE_ROOT:-$HOME/scale-1.6.1-Linux}"
SCALE_LIB="${SCALE_ROOT}/targets/gfx90a/lib"
export ROCR_VISIBLE_DEVICES=0

# Highest working ROCm 7.x for HIP on Frontier (7.13 hipfft is broken for this app)
AMD_VER="${AMD_VER:-7.2.0}"
ROCM_VER="${ROCM_VER:-7.2.0}"
LIBFABRIC_VER="${LIBFABRIC_VER:-}"
ROCM_ROOT="${ROCM_ROOT:-/opt/rocm-${ROCM_VER}}"
ENABLE_TESTING_HIP="${ENABLE_TESTING_HIP:-OFF}"

# Scrub SCALE and stray ROCm installs from the search path for a clean HIP build.
export PATH=$(echo "${PATH}" | tr ':' '\n' | grep -vE 'scale|/opt/rocm-' | paste -sd: -)
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE 'scale|/opt/rocm-' | paste -sd: -)
export LIBRARY_PATH=$(echo "${LIBRARY_PATH:-}" | tr ':' '\n' | grep -vE 'scale|/opt/rocm-' | paste -sd: -)
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH

ml unload darshan-runtime 2>/dev/null || true
if [[ -n "$LIBFABRIC_VER" ]]; then
  ml PrgEnv-amd cray-hdf5-parallel "amd/${AMD_VER}" "rocm/${ROCM_VER}" "libfabric/${LIBFABRIC_VER}" xpmem
else
  ml PrgEnv-amd cray-hdf5-parallel "amd/${AMD_VER}" "rocm/${ROCM_VER}" xpmem
fi
export LIBRARY_PATH=/opt/xpmem/lib64:${LIBRARY_PATH:-}
export PATH="${ROCM_ROOT}/bin:${ROCM_ROOT}/llvm/bin:${PATH}"
export LD_LIBRARY_PATH="${ROCM_ROOT}/lib:${ROCM_ROOT}/llvm/lib:${LD_LIBRARY_PATH:-}"
export ROCM_PATH="${ROCM_PATH:-${ROCM_ROOT}}"

# Link Cray MPI GTL into the CUDA/SCALE binary (needed if experimenting with
# MPICH_GPU_SUPPORT_ENABLED). SCALE multiproc defaults to host-staged MPI.

# Preserve system ROCm for RCCL when SCALE env is sourced later
export FFT_MVEC_ROCM_FOR_RCCL="${ROCM_PATH}"

echo "=== Modules for HIP build ==="
module list 2>&1

# --- Native HIP (clean) ---
rm -rf "${BUILD_HIP}"
if [[ "${ROCM_VER}" == 7.13* ]]; then
  echo "WARNING: rocm/7.13 hipfft fails cufftPlanMany for FFTMatvec; use rocm/7.1.1 instead." >&2
fi
cmake -B "${BUILD_HIP}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_AR="${ROCM_ROOT}/llvm/bin/llvm-ar" \
  -DCMAKE_RANLIB="${ROCM_ROOT}/llvm/bin/llvm-ranlib" \
  -DBUILD_WITH_HIP=ON \
  -DAMDGPU_TARGETS=gfx90a \
  -DENABLE_PROFILING=ON \
  -DFFT_MVEC_ENABLE_NCCL=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DENABLE_TESTING="${ENABLE_TESTING_HIP}"
cmake --build "${BUILD_HIP}" -j

# --- SCALE AMD (clean CUDA build) ---
rm -rf "${BUILD_SCALE}"
source "${SCALE_ROOT}/bin/scaleenv" gfx90a
export PATH="${SCALE_ROOT}/llvm/bin:${SCALE_ROOT}/targets/gfx90a/bin:${PATH}"
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH
export LIBRARY_PATH="${SCALE_LIB}:${LIBRARY_PATH:-}"

# scaleenv gfx90a sets CUDAARCHS=86 — do NOT use 90 (sm_90); that hurts codegen on gfx90a
SCALE_CUDA_ARCH="${CUDAARCHS:-86}"
HOST_CXX=$(command -v amdclang++)

# Use system RCCL so SCALE matches HIP collectives (not host-staged MPI)
export ROCM_PATH="${FFT_MVEC_ROCM_FOR_RCCL}"

cmake -B "${BUILD_SCALE}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${HOST_CXX}" \
  -DCMAKE_CUDA_HOST_COMPILER="${HOST_CXX}" \
  -DCMAKE_EXE_LINKER_FLAGS="-L${SCALE_LIB}" \
  -DENABLE_PROFILING=ON \
  -DFFT_MVEC_ENABLE_NCCL=OFF \
  -DFFT_MVEC_ENABLE_RCCL=OFF \
  -DBUILD_PYTHON_BINDINGS=OFF \
  -DENABLE_TESTING=OFF \
  -DCMAKE_CUDA_COMPILER="${SCALE_ROOT}/targets/gfx90a/bin/nvcc" \
  -DCUDA_ARCH="${SCALE_CUDA_ARCH}" \
  -DCUDAToolkit_ROOT="${SCALE_ROOT}"

cmake --build "${BUILD_SCALE}" -j
echo "Build complete (amd=${AMD_VER} rocm=${ROCM_VER} CUDA_ARCH=${SCALE_CUDA_ARCH}):"
echo "  HIP:   ${BUILD_HIP}/fft_matvec"
echo "  SCALE: ${BUILD_SCALE}/fft_matvec"
