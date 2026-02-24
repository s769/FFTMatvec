import sys
import os
import ctypes

# 1. Guardrail: Check if mpi4py was imported first
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

# 2. Force MPI symbols globally under the hood
sys.setdlopenflags(os.RTLD_NOW | ctypes.RTLD_GLOBAL)

# 3. Import everything from our compiled C++ backend
from ._pyFFTMatvec import *
from ._pyFFTMatvec import Tester
