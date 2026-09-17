# Matrix initialization memory

With the existing `ROW_SETUP=1` default, initialization transforms one local
observation row at a time. The normalized FFT output is scattered directly into
the final `[frequency][column][row]` layout. The two temporary device buffers
hold one padded real row and one complex frequency row, rather than full
matrices. The host input matrix remains fully allocated.

For local dimensions `M` columns, `D` rows, and `T` time samples, the stored
FP64 frequency matrix occupies `16 * M * D * (T + 1)` bytes. Setup scratch is
`8 * M * (2 * T) + 16 * M * (T + 1)` bytes, excluding FFT workspaces and the
persistent execution buffers. Previously, the final layout conversion held
two full frequency matrices simultaneously. For `M=100000`, `D=100`, `T=1000`,
these matrix-related allocations change from 298.32 GiB to 152.14 GiB.

The matvec API, resident coefficient format, precision configuration, and
communication are unchanged. Coefficients are still initialized with a
FP64 FFT before any requested single-precision matrix conversion. A temporary
FP64 plan is created and destroyed when the execution FFT is single precision;
otherwise its existing plan is reused. `ROW_SETUP=0` retains whole-matrix setup.

The synthetic initializers use `size_t` loop counters because their collapsed
OpenMP iteration range can exceed 32 bits at the newly accessible sizes.

## Validation

The new `MatrixSetupTest.NonuniformCoefficientsAndMatvecs` regression compares
primary and auxiliary frequency matrices against a CPU DFT, and F/F-transpose
results against direct CPU convolution. It covers nonuniform coefficients,
singleton axes, odd/even time lengths, and FP64/FP32 execution. It runs as part
of the existing `MatrixTest` target:

```sh
ctest --test-dir build --output-on-failure
mpirun -np 1 build/Tests/MatrixTest --gtest_filter='MatrixSetupTest.*'
mpirun -np 4 build/Tests/MatrixTest --gtest_filter='MatrixSetupTest.*'
mpirun -np 1 compute-sanitizer --tool memcheck --error-exitcode=99 \
  build/Tests/MatrixTest --gtest_filter='MatrixSetupTest.*'
```

Validated on NVIDIA GB200 with CUDA 13.2, GCC 14.3.1, Open MPI 5.0.10,
and NCCL 2.29.3. All six CTest suites passed with four ranks for MPI tests;
the targeted one/four-rank regressions and memcheck/racecheck passed.
Sanitizer runs used Open MPI's `ob1` transport to avoid UCX CUDA-context
initialization diagnostics; the normal suite used the default transport.
HIP was not tested.

At `M=10000`, `D=100`, `T=1000`, one GPU, a linker wrapper tracking explicit
`cudaMalloc`/`cudaFree` calls measured a peak of 30.583 GiB on upstream
`1f773174` and 15.965 GiB with streamed setup. This excludes allocations internal
to shared libraries. Both runs passed the analytical F/F-transpose checks.
The new version also passed those checks at `M=100000`; tracked explicit
allocations peaked at 159.601 GiB. Maximum device usage observed at allocation
checkpoints was 164.3125 GiB, which is not a continuously measured HBM peak.

The regression also passed with `ROW_SETUP=0`. The `INDICES_64_BIT=1` variant
passed after a validation-only fix to an existing `Vector::norm` signed-index
compilation error (`size_t` to `int64_t` for `cublasIdamax_64`); that unrelated
fix is not included here.
