import os
import math
import pytest

# IMPORTANT: pyFFTMatvec must be imported before mpi4py
import pyFFTMatvec
from mpi4py import MPI


@pytest.fixture(scope="module")
def setup_env():
    """Sets up the MPI grid and dimensions for testing."""
    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    # 1D Grid for simplicity in vector testing
    grid_comm = pyFFTMatvec.Comm(comm, 1, size)

    num_blocks = 4
    block_size = 64
    total_elements = size * num_blocks * block_size

    return grid_comm, num_blocks, block_size, total_elements


def get_expected_norm(value, total_elements):
    """Analytically computes the 2-norm of a vector filled with 'value'."""
    return abs(value) * math.sqrt(total_elements)


def test_scalar_inplace_operators(setup_env):
    """Tests +=, -=, *=, /=, and **= against scalars."""
    grid_comm, num_blocks, block_size, total_elements = setup_env

    # Initialize a vector with 1.0s
    v = pyFFTMatvec.Vector(grid_comm, num_blocks, block_size, "col", False, True)
    v.init_vec_ones()

    # Record the raw GPU pointer address
    original_ptr = v.get_d_vec()

    # 1. Addition (+=)
    v += 2.0
    assert math.isclose(v.norm(), get_expected_norm(3.0, total_elements), rel_tol=1e-5)
    assert v.get_d_vec() == original_ptr, "Memory leaked! += allocated a new vector."

    # 2. Subtraction (-=)
    v -= 5.0
    assert math.isclose(v.norm(), get_expected_norm(-2.0, total_elements), rel_tol=1e-5)
    assert v.get_d_vec() == original_ptr, "Memory leaked! -= allocated a new vector."

    # 3. Multiplication (*=)
    v *= -3.0
    assert math.isclose(v.norm(), get_expected_norm(6.0, total_elements), rel_tol=1e-5)
    assert v.get_d_vec() == original_ptr, "Memory leaked! *= allocated a new vector."

    # 4. Division (/=)
    v /= 2.0
    assert math.isclose(v.norm(), get_expected_norm(3.0, total_elements), rel_tol=1e-5)
    assert v.get_d_vec() == original_ptr, "Memory leaked! /= allocated a new vector."

    # 5. Power (**=)
    v **= 2.0
    assert math.isclose(v.norm(), get_expected_norm(9.0, total_elements), rel_tol=1e-5)
    assert v.get_d_vec() == original_ptr, "Memory leaked! **= allocated a new vector."


def test_vector_inplace_operators(setup_env):
    """Tests +=, -=, *=, /= against other vectors."""
    grid_comm, num_blocks, block_size, total_elements = setup_env

    # Initialize v1 with 2.0s
    v1 = pyFFTMatvec.Vector(grid_comm, num_blocks, block_size, "col", False, True)
    v1.init_vec_ones()
    v1.scale(2.0)

    # Initialize v2 with 3.0s
    v2 = pyFFTMatvec.Vector(grid_comm, num_blocks, block_size, "col", False, True)
    v2.init_vec_ones()
    v2.scale(3.0)

    original_ptr = v1.get_d_vec()

    # 1. Addition (+=)
    v1 += v2
    assert math.isclose(v1.norm(), get_expected_norm(5.0, total_elements), rel_tol=1e-5)
    assert v1.get_d_vec() == original_ptr, "Memory leaked! += allocated a new vector."

    # 2. Subtraction (-=)
    v1 -= v2
    assert math.isclose(v1.norm(), get_expected_norm(2.0, total_elements), rel_tol=1e-5)
    assert v1.get_d_vec() == original_ptr, "Memory leaked! -= allocated a new vector."

    # 3. Multiplication (*=)
    v1 *= v2
    assert math.isclose(v1.norm(), get_expected_norm(6.0, total_elements), rel_tol=1e-5)
    assert v1.get_d_vec() == original_ptr, "Memory leaked! *= allocated a new vector."

    # 4. Division (/=)
    v1 /= v2
    assert math.isclose(v1.norm(), get_expected_norm(2.0, total_elements), rel_tol=1e-5)
    assert v1.get_d_vec() == original_ptr, "Memory leaked! /= allocated a new vector."


def test_fused_multiply_add(setup_env):
    """Tests the explicit fused multiply-add methods."""
    grid_comm, num_blocks, block_size, total_elements = setup_env

    v1 = pyFFTMatvec.Vector(grid_comm, num_blocks, block_size, "col", False, True)
    v1.init_vec_ones()
    v1.scale(2.0)  # v1 = 2.0

    v2 = pyFFTMatvec.Vector(grid_comm, num_blocks, block_size, "col", False, True)
    v2.init_vec_ones()
    v2.scale(3.0)  # v2 = 3.0

    v3 = pyFFTMatvec.Vector(grid_comm, num_blocks, block_size, "col", False, True)
    v3.init_vec_ones()
    v3.scale(4.0)  # v3 = 4.0

    original_ptr = v1.get_d_vec()

    # Fused operation: v1 = (v1 * v2) + v3  ->  (2.0 * 3.0) + 4.0 = 10.0
    v1.elementwise_multiply_add_inplace(v2, v3)

    assert math.isclose(
        v1.norm(), get_expected_norm(10.0, total_elements), rel_tol=1e-5
    )
    assert v1.get_d_vec() == original_ptr, "Memory leaked! FMA allocated a new vector."
