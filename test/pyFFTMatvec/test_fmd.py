import os
import math
import torch
import pyFFTMatvec  # MUST BE BEFORE mpi4py!
from mpi4py import MPI


def test_fmd():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    # Set PyTorch GPU device mapping
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    torch.cuda.set_device(local_rank)

    # Replicate the MPI grid math
    proc_cols = int(math.sqrt(world_size))
    proc_rows = world_size // proc_cols
    if proc_rows > proc_cols:
        proc_cols, proc_rows = proc_rows, proc_cols

    grid_comm = pyFFTMatvec.Comm(comm, proc_rows, proc_cols)

    # Resolve Paths (assuming data is in test/data/)
    # We use relpath to go up one directory from test/pyFFTMatvec/ to test/data/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data")

    mat_path = os.path.join(data_path, "test_mat/")
    param_vec_path = os.path.join(data_path, "test_param_vec_SOTI.h5")
    obs_vec_path = os.path.join(data_path, "test_obs_vec_SOTI.h5")

    # Initialize Matrix & Vectors
    F = pyFFTMatvec.Matrix(grid_comm, mat_path)

    m = F.get_vec("input")
    m.init_vec_from_file(param_vec_path)

    d = F.get_vec("output")
    d.init_vec_from_file(obs_vec_path)

    d2 = pyFFTMatvec.Vector(d, False)
    d2.init_vec()

    # Synchronize and Run
    torch.cuda.synchronize()
    comm.Barrier()

    F.matvec(m, d2)

    torch.cuda.synchronize()
    comm.Barrier()

    # Validation
    d2.axpy(-1.0, d)

    norm = d2.norm()
    norm_true = d.norm()

    # Only assert on rank 0, as it holds the global reduced norm
    if grid_comm.get_world_rank() == 0:
        rel_err = norm / norm_true

        # This is all pytest needs! If this fails, pytest catches the AssertionError
        assert rel_err < 1e-6, f"Relative error {rel_err:e} is >= 1e-6"
