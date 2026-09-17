# Reproducing HIP vs SCALE-amd on Frontier (FFTMatvec)

Branch: **`mpi-scale`**  
Machine: OLCF Frontier (MI250X / `gfx90a`)

This note is enough to rebuild both binaries and run the single-GPU HIP vs SCALE timing comparison we used for SCALE feedback. Multi-GPU notes are at the end.

## Prerequisites

1. Clone this repo and check out `mpi-scale`:

```bash
git clone https://github.com/s769/FFTMatvec.git
cd FFTMatvec
git checkout mpi-scale
```

2. Install **SCALE** for AMD (we used `scale-1.6.1-Linux`) under `$HOME/scale-1.6.1-Linux`, or set `SCALE_ROOT` to your install.

3. Optional but recommended: put builds/logs on Lustre (not `$HOME`). The scripts default to `$MEMBERWORK/FFTMatvec` when `MEMBERWORK` is set (OLCF login), otherwise `/lustre/orion/scratch/$USER/FFTMatvec`.

Override as needed:

```bash
export FFT_MVEC_WORK=/lustre/orion/scratch/$USER/<project>/FFTMatvec
export FFT_MVEC_SRC=$HOME/FFTMatvec   # if the clone is elsewhere
export SCALE_ROOT=$HOME/scale-1.6.1-Linux
```

## Modules (login or compute node)

Do **not** use `rocm/7.13` for this app (hipFFT/`cufftPlanMany` issues). Use **7.2.0**:

```bash
ml unload darshan-runtime 2>/dev/null || true
ml PrgEnv-amd cray-hdf5-parallel amd/7.2.0 rocm/7.2.0 xpmem
export LIBRARY_PATH=/opt/xpmem/lib64:${LIBRARY_PATH:-}
```

## Build both binaries

From the repo root (needs an interactive allocation if you prefer building on a compute node; login-node builds are fine):

```bash
# optional: get a debug node
salloc -p batch -q debug -A <YOUR_ACCOUNT> -t 1:00:00 -N 1 -n 8 \
  --network=disable_rdzv_get -c 7 --gpus-per-task 1 --gpu-bind=closest

./scripts/build-frontier-amd.sh
```

That produces:

```text
$FFT_MVEC_WORK/build_hip/fft_matvec     # native HIP (hipify + hipcc)
$FFT_MVEC_WORK/build/fft_matvec         # SCALE CUDA frontend → gfx90a
```

Important SCALE build details (already in the script):

- `source $SCALE_ROOT/bin/scaleenv gfx90a`
- CUDA arch **86** (`CUDAARCHS` from `scaleenv`), not 90
- Keep system `ROCM_PATH` for libraries; put SCALE’s `targets/gfx90a/lib` on `LD_LIBRARY_PATH` at **run** time for the SCALE binary only

## Single-GPU HIP vs SCALE timing (main compare)

Still on a GPU allocation with the modules above:

```bash
export NM=5000          # matrix size parameter; 5000 or 10000 are typical
export NT=1000
export SKIP_SWEEP=1     # skip auto nm search; use NM as set
./scripts/run-frontier-pp.sh
```

What this does:

1. Runs HIP and SCALE each with `srun -n 1 …/fft_matvec -nm $NM -nd 100 -Nt $NT -t -raw`
2. Writes logs under `$FFT_MVEC_WORK/logs/`
3. Builds a Pennycook-style CSV via `scripts/compute_pennycook_fftmatvec.py`

### What “good” looks like (qualitative)

On one MI250X GCD (`ROCR_VISIBLE_DEVICES=0`), **single-rank** HIP and SCALE are usually **within ~1.5–1.7×** on the timed matvec pipeline for `nm=5000` (HIP faster). Both should print:

```text
F Matvec test passed
F* Matvec test passed
```

Example suite PP from one of our Frontier runs (`nm=5000`, ROCm 7.2): SCALE/HIP pipeline ratios ~1.55–1.68 (HIP faster). Re-run for your node; absolute times vary.

### Manual one-liners (if you prefer)

```bash
# HIP
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v scale | paste -sd:)
srun -n 1 -u $FFT_MVEC_WORK/build_hip/fft_matvec -nm 5000 -nd 100 -Nt 1000 -t -raw

# SCALE
export LD_LIBRARY_PATH=$HOME/scale-1.6.1-Linux/targets/gfx90a/lib:$LD_LIBRARY_PATH
srun -n 1 -u $FFT_MVEC_WORK/build/fft_matvec -nm 5000 -nd 100 -Nt 1000 -t -raw
```

## Optional: RedSCALE profiler

```bash
export SLURM_JOB_ID=<your jobid>   # if using an existing salloc
./scripts/profile-frontier-scale.sh
```

Profiles land under `$FFT_MVEC_WORK/profiles/<timestamp>/`.  
Summarize with `$SCALE_ROOT/bin/redscale_profiler summary …`.

Note: for this app the RedSCALE profiler mainly shows pad/unpad/swap kernels; cuBLAS/cuFFT activity is largely **not** attributed (most time looks “unaccounted”). That is itself a useful SCALE-side observation.

## Multi-GPU / multi-rank (important for SCALE)

| Build | Collectives used |
|-------|------------------|
| HIP   | RCCL (device) when available |
| SCALE | **Host-staged MPI** by default |

Reason: Cray MPICH GTL accepts native `hipMalloc` pointers but **rejects SCALE `cudaMalloc` device pointers** (segfault / IPC errors if you force GPU-aware MPI). The `mpi-scale` branch detects this and stages through host memory so multi-rank runs still work.

Expect **large** Bcast/Reduce slowdowns for SCALE vs HIP on 4–8 ranks (orders of magnitude on small messages), even when compute phases are comparable. That is expected with host staging, not a fair “thin wrap” compare for collectives.

Do **not** set `MPICH_GPU_SUPPORT_ENABLED=1` for the SCALE binary unless you are intentionally testing experimental paths (`FFT_MVEC_ASSUME_GPU_AWARE_MPI=1` / `FFT_MVEC_FORCE_HIP_SCRATCH_MPI=1`); those are unsafe/unsupported for production compares.

## Accounts / QOS reminder

Use your project account (`-A …`). Debug QOS has short walls and submit limits; cancel stale interactives if `salloc` fails with `QOSMaxSubmitJobPerUserLimit`.

## Contact / branch contents

`mpi-scale` includes Frontier build/run scripts under `scripts/`, CMake GTL linking for CUDA builds, and SCALE-safe host-staged MPI collectives in `src/gpu_collectives_mpi_cuda.cpp`.
