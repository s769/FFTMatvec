import sys
import os
import ctypes

# 1. Guardrail for MPI import order
if "mpi4py" in sys.modules:
    raise ImportError(
        "\n\n"
        "🔥 IMPORT ORDER ERROR 🔥\n"
        "'mpi4py' was imported before 'pyFFTMatvec'.\n"
        "On HPC systems, pyFFTMatvec MUST be imported first so it can expose \n"
        "global MPI C++ symbols to Python before mpi4py initializes.\n\n"
        "Please rearrange your imports:\n"
        "    import pyFFTMatvec\n"
        "    from mpi4py import MPI\n"
    )

# 2. Force MPI symbols globally
sys.setdlopenflags(os.RTLD_NOW | ctypes.RTLD_GLOBAL)

# 3. Import the C++ backend
from ._pyFFTMatvec import *
from ._pyFFTMatvec import Tester

# ==========================================
# 4. PyTorch Zero-Copy Integration
# ==========================================
import torch


class _CUDAPointer:
    """Internal lightweight CUDA Array Interface wrapper."""

    def __init__(self, ptr, size, dtype_str="<f8"):
        self.ptr = ptr
        self.size = size
        self.dtype_str = dtype_str

    @property
    def __cuda_array_interface__(self):
        return {
            "shape": (self.size,),
            "typestr": self.dtype_str,
            "data": (self.ptr, False),  # False = mutable
            "version": 2,
        }


def _vector_to_torch(self, dtype=torch.float64):
    """
    Converts the FFTMatvec Vector to a zero-copy PyTorch tensor.
    Returns None if the vector is not allocated on the local MPI grid rank.
    """
    if not self.on_grid():
        return None

    size = self.get_num_blocks() * self.get_block_size()
    typestr = "<f8" if dtype == torch.float64 else "<f4"
    ptr = self.get_d_vec()

    # Safety check for uninitialized pointers
    if ptr == 0:
        raise RuntimeError("Cannot wrap a null pointer. Ensure init_vec() was called.")

    cuda_mem = _CUDAPointer(ptr, size, typestr)
    return torch.as_tensor(cuda_mem, device=torch.device("cuda"))


# Dynamically attach the method to the C++ Vector class!
Vector.to_torch = _vector_to_torch
