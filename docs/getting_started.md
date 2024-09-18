## Installation

To build the code, the following dependencies are required:

- CUDA (with cuFFT, cuBLAS, and cuTENSOR 2.x) and a CUDA enabled GPU
- [NCCL](https://github.com/NVIDIA/nccl)
- HDF5

First, clone the repository:
```bash
git clone https://github.com/s769/FFTMatvec.git
cd matvec-test
```

Initialize the submodules:
```bash
git submodule update --init --recursive
```

Then, build the code:
```bash
cmake -B build -DNCCL_LIBRARIES=/path/to/nccl/lib -DNCCL_INCLUDE_DIRS=/path/to/nccl/include -DCMAKE_BUILD_TYPE=Release -DCUTENSOR_ROOT=/path/to/cutensor
cmake --build build
```

**Note**: the `-DCUTENSOR_ROOT` option is only needed if the cuTENSOR 2.x library is not in the usual CUDA library path. Some systems may have the cuTENSOR 1.x library in the CUDA library path, which is not compatible with this code. In that case, the cuTENSOR 2.x library must be [installed](https://developer.nvidia.com/cutensor-downloads), and the path to the cuTENSOR 2.x library must be provided to the build command.

For systems that have compiler wrappers (`cc`, `CC`, `FC`, etc.), use `CC=$(which cc) CXX=$(which CC) FC=$(which ftn) cmake ...`. For some of these systems (e.g. Perlmutter), where the variables such as `MPI_C_LIBRARIES` and `HDF5_INCLUDE_DIRS` are not automatically set, you need to add them manually via
```bash
CC=$(which cc) CXX=$(which CC) FC=$(which ftn) cmake ... -DMPI_INCLUDE=/path/to/mpi/include -DMPI_LIBRARIES=/path/to/mpi/lib/ -DHDF5_INCLUDE_DIRS=/path/to/hdf5/include -DHDF5_LIBRARIES=/path/to/hdf5/lib
```

To build the documentation, the following dependencies are required:

- [MkDocs](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [MkDoxy](https://github.com/JakubAndrysek/MkDoxy)
- [Doxygen](https://www.doxygen.nl/index.html)

Then, build the documentation by running `mkdocs build` or `mkdocs serve`. When running `mkdocs build`, the built documentation will be in the `site` directory.



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
- `-h` (bool): Print help message

**Note**: `pr x pc` must be equal to the number of processors used to run the code. If no values are provided for `-pr` and `-pc`, the code will run with `pr = 1` and `pc = num_mpi_procs`.

For boolean arguments, just pass the flag to enable it without a value. For example:
```bash
mpiexec -np 4 ./build/fft_matvec -pr 2 -pc 2 -g -Nm 20 -Nd 10 -Nt 7 -nm 4 -nd 3 -v -N 100
```

will run the code with 4 processors, a 2x2 processor grid, global sizes, 20 global block columns, 10 global block rows, a block size of 7, 4 local block columns, 3 local block rows, print input/output vectors, and use 100 matvecs for timing.

To reproduce the results in the paper, run with the configurations described in the Numerical Results section.

## Example Program

The following is the main program for the FFTMatvec code (used for testing):

```cpp title="main.cpp"
--8<-- "src/main.cpp"
```

