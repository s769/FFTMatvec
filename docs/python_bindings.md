# pyFFTMatvec

FFTMatvec provides Python bindings via the `pyFFTMatvec` package, built with [pybind11](https://pybind11.readthedocs.io/). The package exposes the core C++ classes (`Comm`, `Vector`, `Matrix`) to Python and includes zero-copy [PyTorch](https://pytorch.org/) GPU integration.

---

## Building the Python Package

### Prerequisites

In addition to the standard [build dependencies](getting_started.md#installation), you need:

- Python 3.8+
- [pybind11](https://pybind11.readthedocs.io/)
- [scikit-build-core](https://scikit-build-core.readthedocs.io/)
- [mpi4py](https://mpi4py.readthedocs.io/)
- [PyTorch](https://pytorch.org/) (for GPU tensor interop)
- [h5py](https://www.h5py.org/) (optional, for creating input data)

### Build with pip

From the repository root:

```bash
pip install .
```

This uses `scikit-build-core` to invoke CMake with the appropriate flags (see `pyproject.toml`). You can customize the build by editing the `[tool.scikit-build]` section.

### Build with CMake directly

Alternatively, build the extension module alongside the C++ code:

```bash
cmake -B build -DBUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release [other flags...]
cmake --build build --parallel
```

This produces the `_pyFFTMatvec` shared library in the build directory.

---

## Import Order

!!! warning "Critical: Import `pyFFTMatvec` before `mpi4py`"

    On HPC systems, `pyFFTMatvec` must be imported **before** `mpi4py` so that global MPI C++ symbols are exposed to Python before `mpi4py` initializes. Importing in the wrong order will raise an `ImportError`.

```python
# ✅ Correct
import pyFFTMatvec
from mpi4py import MPI

# ❌ Wrong — will crash
from mpi4py import MPI
import pyFFTMatvec  # raises ImportError
```

---

## Core Classes

### `Comm` — MPI + GPU Communication

The `Comm` object manages MPI communicators, NCCL communicators, CUDA streams, and cuBLAS handles. It must be created first and passed to all `Matrix` and `Vector` constructors.

```python
import pyFFTMatvec
from mpi4py import MPI

# Create a communicator with a 2×2 processor grid (4 GPUs total)
comm = pyFFTMatvec.Comm(MPI.COMM_WORLD, proc_rows=2, proc_cols=2)

# Query properties
print(f"Rank {comm.get_world_rank()} of {comm.get_world_size()}")
print(f"Grid position: row_color={comm.get_row_color()}, col_color={comm.get_col_color()}")
print(f"GPU device: {comm.get_device()}")
```

### `Matrix` — Block-Triangular Toeplitz Matrix

A `Matrix` can be constructed either from dimensions (for testing) or from a directory path (for real data):

```python
# From dimensions (for testing — initialize with ones or random values)
F = pyFFTMatvec.Matrix(comm, cols=20, rows=10, block_size=50, global_sizes=True)
F.init_mat_ones()        # fill with ones
# or
F.init_mat_doubles()     # fill with deterministic random doubles

# From a directory path (reads meta_adj and HDF5 files — see I/O Format)
F = pyFFTMatvec.Matrix(comm, path="/path/to/my_matrix")

# With an auxiliary matrix
F = pyFFTMatvec.Matrix(comm, path="/path/to/my_matrix", aux_path="/path/to/aux_matrix")

# With mixed precision
p_config = pyFFTMatvec.MatvecPrecisionConfig()
p_config.fft = pyFFTMatvec.Precision.SINGLE
p_config.sbgemv = pyFFTMatvec.Precision.SINGLE
F = pyFFTMatvec.Matrix(comm, path="/path/to/my_matrix", p_config=p_config)
```

### `Vector` — Distributed Block Vector

Vectors are distributed across MPI ranks. A "column" vector lives in the parameter space, and a "row" vector lives in the observation space.

```python
# Create vectors from a matrix (recommended — dimensions match automatically)
x = F.get_vec("input")     # column vector (Nm * Nt)
y = F.get_vec("output")    # row vector (Nd * Nt)

# Or create directly with explicit dimensions
x = pyFFTMatvec.Vector(comm, blocks=20, block_size=50, row_or_col="col", global_sizes=True)

# Initialize
x.init_vec_ones()          # fill with 1.0
x.init_vec_zeros()         # fill with 0.0
x.init_vec_doubles()       # fill with deterministic random doubles
x.init_vec_consecutive()   # fill with 0, 1, 2, ...
x.init_vec()               # mark as initialized (use existing GPU memory)

# Query properties
print(f"Local blocks: {x.get_num_blocks()}, Global blocks: {x.get_glob_num_blocks()}")
print(f"Block size: {x.get_block_size()}, On grid: {x.on_grid()}")
```

### `MatvecPrecisionConfig` — Mixed Precision

Controls the precision of each stage of the FFT-based matvec algorithm:

```python
p_config = pyFFTMatvec.MatvecPrecisionConfig()
p_config.broadcast_and_pad = pyFFTMatvec.Precision.DOUBLE   # default
p_config.fft              = pyFFTMatvec.Precision.SINGLE    # use single-precision FFT
p_config.sbgemv           = pyFFTMatvec.Precision.SINGLE    # use single-precision GEMV
p_config.ifft             = pyFFTMatvec.Precision.SINGLE    # use single-precision IFFT
p_config.unpad_and_reduce = pyFFTMatvec.Precision.DOUBLE    # default
```

---

## Matrix-Vector Operations

```python
# Forward matvec: y = F @ x
F.matvec(x, y)

# Transpose matvec: y = F^T @ x
F.transpose_matvec(x, y)

# Using auxiliary matrix G: y = G @ x
F.matvec(x, y, use_aux_mat=True)

# Full matvec (F^T F): y = F^T F @ x
F.matvec(x, y, full=True)
```

---

## Vector Arithmetic

`Vector` supports standard Python arithmetic operators. All operations are executed on the GPU.

### Scalar Operations

```python
# Arithmetic with scalars
z = x * 2.0          # scale
z = x + 1.0          # add scalar
z = x - 0.5          # subtract scalar
z = x / 3.0          # divide by scalar
z = 1.0 / x          # element-wise reciprocal
z = x ** 2.0         # element-wise power

# In-place variants
x *= 2.0
x += 1.0
x -= 0.5
x /= 3.0
x **= 2.0
```

### Vector-Vector Operations

```python
# Element-wise operations
z = x + y
z = x - y
z = x * y             # element-wise multiply
z = x / y             # element-wise divide

# In-place variants
x += y
x -= y
x *= y
x /= y

# Dot product and norms
dot_val = x.dot(y)
l2_norm = x.norm(2)       # L2 norm (default)
l1_norm = x.norm(1)       # L1 norm
linf    = x.norm(0)       # L-infinity norm

# BLAS-style operations
x.scale(2.0)                     # x = 2.0 * x (in-place)
x.axpy(alpha, y)                 # x = x + alpha * y (in-place)
z = x.waxpy(alpha, y)            # z = x + alpha * y (returns new)
x.axpby(alpha, beta, y)          # x = alpha * x + beta * y (in-place)
z = x.waxpby(alpha, beta, y)     # z = alpha * x + beta * y (returns new)
```

### Additional Operations

```python
# Element-wise operations (explicit method calls)
z = x.elementwise_multiply(y)
x.elementwise_multiply_inplace(y)
z = x.elementwise_divide(y)
x.elementwise_divide_inplace(y)
z = x.elementwise_inverse()          # z_i = 1 / x_i
x.elementwise_inverse_inplace()
z = x.pow(0.5)                       # z_i = x_i^0.5
x.pow_inplace(0.5)
z = x.add_scalar(1.0)
x.add_scalar_inplace(1.0)

# Fused multiply-add: z_i = x_i * y_i + z_i
z = x.elementwise_multiply_add(y, z)
x.elementwise_multiply_add_inplace(y, z)

# Copy
x.copy_to(y)             # y ← x

# Resize
z = x.extend(new_block_size)   # extend block size (zero-padded)
z = x.shrink(new_block_size)   # shrink block size (truncated)
z = x.resize(new_block_size)   # extend or shrink as needed
```

---

## File I/O

See the [I/O and Data Formats](io_format.md) page for details on the file format.

### Reading Vectors

```python
# Read a column vector from an HDF5 file
x = pyFFTMatvec.Vector(comm, blocks=20, block_size=50, row_or_col="col", global_sizes=True)
x.init_vec_from_file("my_vector.h5")

# With checksum verification
x.init_vec_from_file("my_vector.h5", checksum=42)

# For QoI vectors
x.init_vec_from_file("qoi_vector.h5", QoI=True)
```

### Writing Vectors

```python
# Save to HDF5
y.save("output.h5")

# Save as QoI vector (adds qoi=1 attribute)
y.save("qoi_output.h5", QoI=True)
```

---

## PyTorch Integration

`pyFFTMatvec` provides **zero-copy** interop between `Vector` objects and PyTorch tensors via the CUDA Array Interface. No data is copied between CPU and GPU — both the `Vector` and the PyTorch tensor share the same GPU memory.

### Vector → PyTorch Tensor

```python
import torch

# Get a zero-copy PyTorch view of the GPU data
t = x.to_torch()                         # default: float64
t = x.to_torch(dtype=torch.float32)      # cast to float32

# t is a regular PyTorch CUDA tensor — use it in any PyTorch computation
result = torch.nn.functional.relu(t)
loss = (t ** 2).sum()
```

!!! note
    `to_torch()` returns `None` if the calling MPI rank does not own data for this vector (i.e., `on_grid()` returns `False`).

### PyTorch Tensor → Vector

```python
# Copy data from a PyTorch tensor into the Vector's GPU memory
tensor = torch.randn(x.get_num_blocks() * x.get_block_size(), device="cuda")
x.from_torch(tensor)
```

The tensor must:

- Be on the GPU (`device="cuda"`)
- Have the same total number of elements as the vector
- Multi-dimensional tensors are automatically flattened if the total element count matches

### Complete Example

```python
import pyFFTMatvec
from mpi4py import MPI
import torch

comm = pyFFTMatvec.Comm(MPI.COMM_WORLD, proc_rows=1, proc_cols=2)

# Load matrix and create vectors
F = pyFFTMatvec.Matrix(comm, path="/path/to/matrix")
x = F.get_vec("input")
y = F.get_vec("output")

# Initialize x from a PyTorch tensor
if x.on_grid():
    size = x.get_num_blocks() * x.get_block_size()
    t = torch.randn(size, device="cuda", dtype=torch.float64)
    x.from_torch(t)
    x.init_vec()

# Compute matvec
F.matvec(x, y)

# Use the result in PyTorch
y_torch = y.to_torch()
if y_torch is not None:
    loss = (y_torch ** 2).sum()
    loss.backward()   # gradients flow through the shared memory
```

---

## Testing Utilities

The `Tester` submodule provides verification functions:

```python
# Check that a ones-matrix matvec produces the expected result
pyFFTMatvec.Tester.check_ones_matvec(comm, F, out_vec, conj=False, full=False)
# conj=True for transpose matvec, full=True for full (F^T F) matvec
```

This checks that $F \cdot \mathbf{1} = \text{expected}$ when $F$ is initialized with `init_mat_ones()`.
