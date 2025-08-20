# Multi-GPU Accelerated FFT-Based Matvec for Block Triangular Toeplitz Matrices

This repository contains the code for the paper "Sreeram Venkat, Milinda Fernando, Stefan Henneking, and Omar Ghattas. _Fast and Scalable FFT-Based GPU-Accelerated Algorithms for Hessian Actions Arising in Linear Inverse Problems Governed by Autonomous Dynamical Systems_. arXiv preprint [arXiv:2407.13066](https://arxiv.org/abs/2407.13066). 2024 Jul 18."

This branch enables performance portability to AMD GPUs and mixed-precision computations. See "Sreeram Venkat, Kasia Swirydowicz, Noah Wolfe, and Omar Ghattas. _Mixed-Precision Performance Portability of FFT-Based GPU-Accelerated Algorithms for Block-Triangular Toeplitz Matrices_. arXiv preprint [arXiv:2508.10202] (https://arxiv.org/abs/2508.10202) 2025 Aug 10."

## Documentation

The documentation for the code can be found [here](https://fftmatvec.readthedocs.io/en/latest/). That page has the documentation for the `main` branch. This is the `mp` branch which supports AMD GPUs via HIP and mixed-precision computing. 

## Installation

To build the code, the following dependencies are required:

- CUDA (with cuFFT, cuBLAS, and (optionally) cuTENSOR 2.x) and a CUDA enabled GPU OR ROCm/HIP (with rocFFT, rocBLAS, and RCCL) and an AMD GPU
- [NCCL](https://github.com/NVIDIA/nccl) for running on NVIDIA GPUs
- HDF5 (parallel version is required)

First, clone the repository:
```bash
git clone https://github.com/s769/FFTMatvec.git
git checkout mp
```

Then, build the code (CUDA):
```bash
cmake -B build -DNCCL_LIBRARIES=/path/to/nccl/lib -DNCCL_INCLUDE_DIRS=/path/to/nccl/include -DCMAKE_BUILD_TYPE=Release [-DCUTENSOR_ROOT=/path/to/cutensor] [-DCUDA_ARCH=XX]
cmake --build build --parallel
```
or (HIP)
```
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_HIP=ON [-DAMDGPU_TARGETS=gfxABC]
```

**Note**: the `-DCUTENSOR_ROOT` option is only needed if the cuTENSOR 2.x library is not in the usual CUDA library path. Some systems may have the cuTENSOR 1.x library in the CUDA library path, which is not compatible with this code. In that case, the cuTENSOR 2.x library must be [installed](https://developer.nvidia.com/cutensor-downloads), and the path to the cuTENSOR 2.x library must be provided to the build command.

Tests will build by default. If you don't want to build the tests, you can disable them by adding `-DENABLE_TESTING=OFF` to the cmake command. To run the tests, use `ctest` in the `build` directory. Tests require a minimum of 2 GPUs to run.


To build the documentation, the following dependencies are required:

- [MkDocs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MkDoxy](https://github.com/JakubAndrysek/MkDoxy)
- [Doxygen](https://www.doxygen.nl/index.html)

Then, build the documentation by running `mkdocs build` or `mkdocs serve` in the `docs` directory. The built documentation will be in the `site` directory.


## Usage

The main executable is `fft_matvec`. It takes the following arguments:

- `-pr` (int): Number of processor rows (default: 1)
- `-pc` (int): Number of processor columns (default: 1)
- `-g` (bool): Use global sizes (default: false)
- `-Nm` (int): Number of global block columns (default: 10, ignored if `-g` is false)
- `-Nd` (int): Number of global block rows (default: 5, ignored if `-g` is false)   
- `-Nt` (int): Block size (default: 7)
- `-nm` (int): Number of local block columns (default: 3, ignored if `-g` is true)
- `-nd` (int): Number of local block rows (default: 2, ignored if `-g` is true)
- `-v` (bool): Print input/output vectors (default: false)
- `-N` (int): Number of matvecs to use for timing (default: 100)
- `-raw` (bool): Print raw timing data instead of table (default: false)
- `-t` (bool): Check matvec results (default: false)
- `-prec` (string): Precision Code: 5 characters, each D or S (case insensitive) representing the precision of the corresponding matrix/vector component (D=double, S=single). Components are: broadcast/pad, fft, sbgemv, ifft, unpad/reduce. Default is DDDDD.
- `-s` (string): Directory to save output files to (default "" - don't save output)
- `-rand` (bool): Use deterministic double precision values for the input vectors/matrices that cannot be represented as single precision floats without error. Result checking will not work with this option enabled (default: false)
- `-h` (bool): Print help message

**Note**: `pr x pc` must be equal to the number of processors used to run the code. If no values are provided for `-pr` and `-pc`, the code will run with `pr = 1` and `pc = num_mpi_procs`.

For boolean arguments, just pass the flag to enable it without a value. For example:
```bash
mpiexec -np 4 ./build/fft_matvec -pr 2 -pc 2 -g -Nm 20 -Nd 10 -Nt 7 -nm 4 -nd 3 -v -N 100 -prec dssdd -rand -s .
```

will run the code with 4 processors, a 2x2 processor grid, global sizes, 20 global block columns, 10 global block rows, a block size of 7, 4 local block columns, 3 local block rows, print input/output vectors, and use 100 matvecs for timing. The fft and sbgemv will be computed in single precision; all other components in double precision. Deterministic double precision values that cannot be represented as single precision floats without error will be used to initialize the matrix/vector, and output vectors will be saved in the current directory.

To reproduce the results in the paper, run with the configurations described in the Numerical Results section.


## License

This code is released under the MIT License. See [LICENSE](LICENSE) for more information.






