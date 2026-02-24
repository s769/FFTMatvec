import os
import torch
import pyFFTMatvec  # MUST BE BEFORE mpi4py!
from mpi4py import MPI


class CUDAPointer:
    """
    A lightweight object that implements the CUDA Array Interface.
    PyTorch uses this to wrap raw device memory without copying it.
    """

    def __init__(self, ptr, size, dtype_str="<f8"):
        self.ptr = ptr
        self.size = size
        self.dtype_str = dtype_str

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.size,),
            "typestr": self.dtype_str,
            "data": (self.ptr, False),
            "version": 2,
        }


def wrap_ptr_in_pytorch(ptr, size, dtype=torch.float64):
    """
    Wraps a raw device pointer in a PyTorch tensor without taking ownership.
    """
    if dtype == torch.float64:
        typestr = "<f8"
    elif dtype == torch.float32:
        typestr = "<f4"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    cuda_mem = CUDAPointer(ptr, size, typestr)
    return torch.as_tensor(cuda_mem, device=torch.device("cuda"))


def test_ones_matvec():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank)

    # Process grid dimensions (using a 1D grid for this test)
    proc_rows = 1
    proc_cols = size

    grid_comm = pyFFTMatvec.Comm(proc_rows, proc_cols)

    # Setup Configuration
    p_config = pyFFTMatvec.MatvecPrecisionConfig()
    p_config.fft = pyFFTMatvec.Precision.DOUBLE
    p_config.sbgemv = pyFFTMatvec.Precision.DOUBLE

    # Problem Dimensions
    num_col_blocks = 4
    num_row_blocks = 4
    block_size = 64

    # Initialize Matrix and Vectors
    mat = pyFFTMatvec.Matrix(
        grid_comm, num_col_blocks, num_row_blocks, block_size, False, False, p_config
    )
    mat.init_mat_ones()

    x = pyFFTMatvec.Vector(grid_comm, num_col_blocks, block_size, "col", False, True)
    y = pyFFTMatvec.Vector(grid_comm, num_row_blocks, block_size, "row", False, True)

    x.init_vec_ones()
    y.init_vec_zeros()

    # Wrap device pointers and initialize ONLY if they exist locally
    if x.on_grid():
        local_x_size = x.get_num_blocks() * x.get_block_size()
        pt_x = wrap_ptr_in_pytorch(x.get_d_vec(), local_x_size, dtype=torch.float64)
        pt_x.fill_(1.0)

    if y.on_grid():
        local_y_size = y.get_num_blocks() * y.get_block_size()
        pt_y = wrap_ptr_in_pytorch(y.get_d_vec(), local_y_size, dtype=torch.float64)
        pt_y.fill_(0.0)

    # Synchronize before execution to ensure data is ready
    torch.cuda.synchronize()
    comm.Barrier()

    # Execute the C++ Matvec (it handles internal MPI safety)
    mat.matvec(x, y)

    # Synchronize after execution
    torch.cuda.synchronize()
    comm.Barrier()

    # Validate using the C++ internal tester
    conj = False
    full = False

    # If this fails, the C++ assertion will crash the test natively
    pyFFTMatvec.Tester.check_ones_matvec(grid_comm, mat, y, conj, full)

    # Extra Python-side validation just to ensure the PyTorch tensors reflect the C++ memory
    if y.on_grid():
        norm_y = torch.linalg.norm(pt_y)
        assert norm_y.item() > 0.0, (
            "PyTorch tensor norm is zero; memory mapping failed."
        )
