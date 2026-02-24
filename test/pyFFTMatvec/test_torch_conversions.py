import os
import math
import torch
import pyFFTMatvec  # MUST BE BEFORE mpi4py!
from mpi4py import MPI


def test_torch_roundtrip():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Set PyTorch GPU device mapping
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank)

    # Process grid dimensions (using a 1D grid for this test)
    grid_comm = pyFFTMatvec.Comm(comm, 1, size)

    # Problem Dimensions
    num_blocks = 4
    block_size = 64

    # 1. Initialize C++ Vector with 1.0s
    x = pyFFTMatvec.Vector(grid_comm, num_blocks, block_size, "col", False, True)
    x.init_vec_ones()

    # We only perform ML ops on ranks that actually hold part of the vector
    if x.on_grid():
        # 2. Extract zero-copy view FROM C++
        pt_x = x.to_torch()

        # Verify extraction worked
        assert torch.allclose(pt_x, torch.ones_like(pt_x)), (
            "C++ -> PyTorch extraction failed!"
        )

        # 3. Simulate an ML layer
        # By multiplying and adding, PyTorch allocates a BRAND NEW tensor
        # in a different part of the GPU memory.
        ml_output = (pt_x * 3.0) + 2.0  # Every element should now be 5.0

        # 4. Push the new ML data BACK into the C++ memory
        x.from_torch(ml_output)

        # 5. Local Validation:
        # Because pt_x is a zero-copy window into the C++ memory, checking pt_x
        # instantly verifies if the C++ memory was successfully overwritten
        assert torch.allclose(pt_x, torch.full_like(pt_x, 5.0)), (
            "PyTorch -> C++ insertion failed!"
        )

    # Synchronize across the MPI grid before running C++ reductions
    torch.cuda.synchronize()
    comm.Barrier()

    # 6. Global Validation: Use the native C++ math library to compute the norm
    cpp_norm = x.norm()

    # Only Rank 0 asserts on the global norm
    if rank == 0:
        total_elements = size * num_blocks * block_size
        expected_norm = math.sqrt(total_elements * (5.0**2))

        # If this fails, pytest automatically catches it and reports the mismatch
        assert math.isclose(cpp_norm, expected_norm, rel_tol=1e-5), (
            f"Norm mismatch! Expected {expected_norm:.4f}, got {cpp_norm:.4f}"
        )
