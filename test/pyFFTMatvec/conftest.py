import os
import sys

# Remember our golden rule: import the bindings before MPI!
import pyFFTMatvec
from mpi4py import MPI


def pytest_configure(config):
    """
    This hook runs before any tests are collected or executed.
    We use it to silence the pytest output on all non-zero MPI ranks.
    """
    comm = MPI.COMM_WORLD
    if comm.Get_rank() != 0:
        # Block pytest's terminal reporter so it doesn't print the test
        # summaries, progress dots, or collection statuses.
        config.pluginmanager.set_blocked("terminalreporter")
