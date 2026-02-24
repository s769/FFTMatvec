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


def _vector_from_torch(self, tensor):
    """
    Safely copies data from a PyTorch tensor into the FFTMatvec Vector's memory.
    The input tensor must be on the GPU and match the vector's size and dtype.
    """
    if not self.on_grid():
        return

    # Grab the zero-copy view of our C++ memory
    view = self.to_torch(tensor.dtype)

    if view is None:
        return

    # Ensure sizes match so we don't cause a CUDA segfault
    if view.shape != tensor.shape:
        # If the tensor is multidimensional (e.g., [batch, features]),
        # try flattening it to see if the total elements match.
        if view.numel() == tensor.numel():
            tensor = tensor.view(-1)
        else:
            raise ValueError(
                f"Shape mismatch: FFTMatvec Vector expects {view.numel()} elements, "
                f"but PyTorch tensor has {tensor.numel()}.",
            )

    # Perform a highly optimized CUDA device-to-device copy
    # This safely copies the data FROM the PyTorch tensor INTO the C++ memory
    view.copy_(tensor)


# Attach both methods to the C++ Vector class
Vector.to_torch = _vector_to_torch
Vector.from_torch = _vector_from_torch
