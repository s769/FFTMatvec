# I/O Data Formats

FFTMatvec uses [HDF5](https://www.hdfgroup.org/solutions/hdf5/) (via the [HighFive](https://github.com/BlueBrain/HighFive) C++ library) for all on-disk storage. Parallel HDF5 with MPI collective I/O is used for both reading and writing, so the same files work seamlessly across any number of MPI ranks.

This page describes how matrix directories and vector files should be formatted so that FFTMatvec can read them, and what FFTMatvec produces when it writes output.

---

## Terminology

Before diving in, here is a quick glossary of the key dimensions:

| Symbol | Meaning |
|--------|---------|
| `Nd` | Global number of **block rows** (observation blocks) |
| `Nm` | Global number of **block columns** (parameter blocks) |
| `Nt` | **Block size** (number of time steps per block) |

A block-triangular Toeplitz (BTT) matrix $F$ has shape $(N_d \cdot N_t) \times (N_m \cdot N_t)$. It is stored as $N_d$ block rows, each of length $N_m \cdot N_t$.

---

## Data Ordering (SOTI)

FFTMatvec uses **SOTI (Sensors/parameters-Ordered-Then-Indexed)** ordering for all data. In this convention, data is stored with the block (sensor/parameter) index as the outer dimension and the time-step index as the inner dimension. Concretely, a vector $\mathbf{v}$ of $N$ blocks of size $N_t$ is stored as:

$$
[\underbrace{v_{0,0}, v_{0,1}, \ldots, v_{0,N_t-1}}_{\text{block 0}},\; \underbrace{v_{1,0}, v_{1,1}, \ldots, v_{1,N_t-1}}_{\text{block 1}},\; \ldots]
$$

This is indicated by the `reindex` attribute in HDF5 datasets. FFTMatvec **requires** `reindex = 1` (i.e., SOTI ordering) for all input data.

---

## Matrix Directory Structure

A matrix is stored as a directory with a specific layout. When you construct a `Matrix` from a path (either in C++ or Python), the code expects the following structure:

```
my_matrix/
└── binary/
    ├── meta_adj              ← plain-text metadata file
    ├── <prefix>000000.h5     ← block row 0
    ├── <prefix>000001.h5     ← block row 1
    ├── <prefix>000002.h5     ← block row 2
    └── ...                   ← one file per block row
```

### The `meta_adj` File

The `meta_adj` file is a simple plain-text file with **nine lines**, each containing a single value. The format is:

```
<global_num_rows>       (line 0: Nd — number of block rows)
<global_num_cols>       (line 1: Nm — number of block columns)
<block_size>            (line 2: Nt — block size / time steps)
<prefix>                (line 3: filename prefix string, e.g. "F_")
<extension>             (line 4: file extension, must be ".h5")
<reindexed>             (line 5: must be 1)
<is_p2q>                (line 6: 0 for parameter-to-observable, 1 for parameter-to-QoI)
<reverse_dof>           (line 7: must be 1)
<checksum>              (line 8: integer checksum, 0 means no checksum verification)
```

**Example** `meta_adj` for a matrix with 10 block rows, 20 block columns, block size 50:

```
10
20
50
F_
.h5
1
0
1
0
```

### Block-Row HDF5 Files

Each block row is stored in a separate HDF5 file named `<prefix><NNNNNN>.h5`, where `NNNNNN` is the zero-padded (6-digit) block-row index. For example, with prefix `F_`, the files would be `F_000000.h5`, `F_000001.h5`, etc.

Each HDF5 file contains:

- **Dataset** `"vec"` — a 1D array of doubles with shape `(Nm * Nt,)`, stored in SOTI ordering. This is the flattened block row: block column 0's $N_t$ values, then block column 1's, and so on.
- **Attributes** on the `"vec"` dataset:
    - `reindex` (int) — must be `1`
    - `n_param` (int) — must equal `Nm` (global number of block columns)
    - `param_steps` (int) — must equal `Nt` (block size)
    - `checksum` (int, optional) — if the meta file specifies a nonzero checksum, this must match

---

## Vector HDF5 Format

A vector is stored as a single HDF5 file. Column vectors and row vectors use slightly different attribute names but share the same dataset structure.

### Dataset

- **Dataset** `"vec"` — a 1D array of doubles with shape `(glob_num_blocks * block_size,)`, stored contiguously in SOTI ordering.

### Attributes

The attributes on the `"vec"` dataset depend on whether the vector is a **column vector** (parameter-space, `row_or_col = "col"`) or a **row vector** (observation-space, `row_or_col = "row"`):

**Column vectors** (parameter-space):

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_param` | int | Global number of blocks (`Nm`) |
| `param_steps` | int | Block size (`Nt`) |
| `reindex` | int | Must be `1` (SOTI ordering) |
| `checksum` | int | Optional — present only if checksum ≠ 0 |

**Row vectors** (observation-space):

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_obs` | int | Global number of blocks (`Nd`) |
| `obs_steps` | int | Block size (`Nt`) |
| `reindex` | int | Must be `1` (SOTI ordering) |
| `qoi` | int | Optional — `1` if this is a QoI (quantity-of-interest) vector |

---

## Worked Example

Consider a small matrix with:

- `Nd = 3` block rows
- `Nm = 4` block columns
- `Nt = 5` block size
- prefix `F_`

### Directory layout

```
my_matrix/
└── binary/
    ├── meta_adj
    ├── F_000000.h5
    ├── F_000001.h5
    └── F_000002.h5
```

### Contents of `meta_adj`

```
3
4
5
F_
.h5
1
0
1
0
```

### Each HDF5 file

Each file (e.g., `F_000000.h5`) contains:

- Dataset `"vec"`: shape `(20,)` — that is $N_m \times N_t = 4 \times 5 = 20$ doubles
- Attributes: `reindex = 1`, `n_param = 4`, `param_steps = 5`

### Corresponding vectors

- A **column (input) vector** for this matrix has shape `(20,)` ($N_m \times N_t$) with attributes `n_param = 4`, `param_steps = 5`, `reindex = 1`
- A **row (output) vector** has shape `(15,)` ($N_d \times N_t$) with attributes `n_obs = 3`, `obs_steps = 5`, `reindex = 1`

---

## Creating Matrix Data with Python (h5py)

If you are generating matrix data outside of FFTMatvec (e.g., from a PDE solver), you can create the directory structure using Python and [h5py](https://www.h5py.org/):

```python
import h5py
import numpy as np
import os

Nd = 3   # block rows
Nm = 4   # block columns
Nt = 5   # block size
prefix = "F_"
out_dir = "my_matrix/binary"
os.makedirs(out_dir, exist_ok=True)

# Write meta_adj
with open(os.path.join(out_dir, "meta_adj"), "w") as f:
    f.write(f"{Nd}\n{Nm}\n{Nt}\n{prefix}\n.h5\n1\n0\n1\n0\n")

# Write each block row
for r in range(Nd):
    filename = os.path.join(out_dir, f"{prefix}{r:06d}.h5")
    data = np.random.randn(Nm * Nt)   # replace with your actual data

    with h5py.File(filename, "w") as f:
        ds = f.create_dataset("vec", data=data)
        ds.attrs["reindex"] = 1
        ds.attrs["n_param"] = Nm
        ds.attrs["param_steps"] = Nt
```

Similarly, to create a vector file:

```python
import h5py
import numpy as np

Nm = 4
Nt = 5
data = np.random.randn(Nm * Nt)

with h5py.File("my_vector.h5", "w") as f:
    ds = f.create_dataset("vec", data=data)
    ds.attrs["reindex"] = 1
    ds.attrs["n_param"] = Nm      # use "n_obs" for row vectors
    ds.attrs["param_steps"] = Nt  # use "obs_steps" for row vectors
```

---

## Reading and Writing in C++

### Reading a Matrix from Disk

```cpp
// Construct a Matrix from a directory path
// The code reads binary/meta_adj and binary/<prefix>NNNNNN.h5 files
Matrix F(comm, "/path/to/my_matrix");

// Optionally load an auxiliary matrix (must share the same dimensions)
Matrix F_with_aux(comm, "/path/to/my_matrix", "/path/to/aux_matrix");
```

### Reading a Vector from Disk

```cpp
// Create a vector with matching dimensions, then read from file
Vector v(comm, Nm, Nt, "col", true);      // column vector with global sizes
v.init_vec_from_file("my_vector.h5");

// With checksum verification
v.init_vec_from_file("my_vector.h5", 42); // expects checksum == 42
```

### Writing a Vector to Disk

```cpp
// After computing a result, save it
result_vec.save("output.h5");

// For QoI vectors, pass QoI = true
qoi_vec.save("qoi_output.h5", true);
```

---

## Reading and Writing in Python

### Reading a Matrix

```python
import pyFFTMatvec
from mpi4py import MPI

comm = pyFFTMatvec.Comm(MPI.COMM_WORLD, proc_rows, proc_cols)

# Load matrix from directory
F = pyFFTMatvec.Matrix(comm, path="/path/to/my_matrix")

# With mixed precision
p_config = pyFFTMatvec.MatvecPrecisionConfig()
p_config.fft = pyFFTMatvec.Precision.SINGLE
F = pyFFTMatvec.Matrix(comm, path="/path/to/my_matrix", p_config=p_config)
```

### Reading and Writing Vectors

```python
# Create matching vectors from the matrix
x = F.get_vec("input")    # column vector with correct dimensions
y = F.get_vec("output")   # row vector with correct dimensions

# Read vector from file
x.init_vec_from_file("my_vector.h5")

# Compute matvec
F.matvec(x, y)

# Save result
y.save("result.h5")
```

---

## Checksum Verification

FFTMatvec supports optional checksum verification to ensure data consistency. If the `meta_adj` file specifies a nonzero checksum (line 8), then:

1. Each block-row HDF5 file must have a `checksum` attribute on its `"vec"` dataset matching the value in `meta_adj`
2. Vector files loaded via `init_vec_from_file` can also be checked by passing a `checksum` argument

This is useful when working with large datasets across different storage systems to verify data integrity.

## Auxiliary Matrices

FFTMatvec supports loading an **auxiliary matrix** $G$ alongside the primary matrix $F$. This is used in the context of computing the full product $G^T F$ (or similar operations). The auxiliary matrix must:

- Have the **same dimensions** (`Nd`, `Nm`, `Nt`) as the primary matrix
- Have the **same checksum** as the primary matrix
- Be stored in the same directory format as the primary matrix

You load it by providing the `aux_path` argument when constructing a `Matrix`.
