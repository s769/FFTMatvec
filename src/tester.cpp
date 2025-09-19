
#include "tester.hpp"

// Function to check if two double values are equal within a tolerance
bool double_equality(double a, double b, double tol)
{
    return (std::abs(a - b) <= tol * std::max(1.0, std::max(std::abs(a), std::abs(b))));
}

// Function to check if an element of the matrix-vector product with ones matrix and ones vector is
// correct
void check_element(
    Comm& comm, double elem, size_t b, size_t j, size_t Nt, size_t Nm, size_t Nd, bool conj, bool full, double tol)
{
    // use double_equality to check for equality
    double correct_elem;
    if (conj) {
        if (full) {
            correct_elem = (Nm * Nd * ((j + 1) * (2 * Nt - j))) / 2.0;
        } else {
            correct_elem = (Nt - j) * Nd;
        }
    } else {
        if (full) {
            correct_elem = (Nm * Nd * ((Nt - j) * (2 * (j + 1) + (Nt - j - 1)))) / 2.0;
        } else {
            correct_elem = (j + 1) * Nm;
        }
    }
    if (!double_equality(elem, correct_elem, tol)) {
        int row_color = comm.get_row_color();
        int col_color = comm.get_col_color();
        std::cerr << "Error: check_element: incorrect element in process (" << row_color << ", "
                  << col_color << ") block: " << b << ", t: " << j << std::endl;
        std::cerr << "Expected: " << correct_elem << ", got: " << elem << std::endl;
        MPICHECK(MPI_Abort(MPI_COMM_WORLD, 1));
    }
}

void Tester::check_ones_matvec(Comm& comm, Matrix& mat, Vector& vec, bool conj, bool full)
{

    int proc_rows = comm.get_proc_rows();
    int proc_cols = comm.get_proc_cols();

    int Nt = mat.get_block_size();

    int Nm = mat.get_glob_num_cols();
    int Nd = mat.get_glob_num_rows();

    MatvecPrecisionConfig precision_config = mat.get_precision_config();
    double tol = 1e-6;
    if (precision_config.broadcast_and_pad == Precision::SINGLE ||
        precision_config.fft == Precision::SINGLE ||
        precision_config.sbgemv == Precision::SINGLE ||
        precision_config.ifft == Precision::SINGLE ||
        precision_config.unpad_and_reduce == Precision::SINGLE) {
        tol = 1e-1;
    }

    if (vec.on_grid()) {
        double* d_vec = vec.get_d_vec();
        int num_blocks = vec.get_num_blocks();
        double* h_vec = new double[num_blocks * Nt];

        gpuErrchk(
            cudaMemcpy(h_vec, d_vec, num_blocks * Nt * sizeof(double), cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < num_blocks; i++) {
            for (size_t j = 0; j < Nt; j++) {
                check_element(comm, h_vec[i * Nt + j], i, j, Nt, Nm, Nd, conj, full, tol);
            }
        }

        delete[] h_vec;
    }

    std::string matvec_name;
    if (conj) {
        if (full) {
            matvec_name = "FF*";
        } else {
            matvec_name = "F*";
        }
    } else {
        if (full) {
            matvec_name = "F*F";
        } else {
            matvec_name = "F";
        }
    }
    if (comm.get_world_rank() == 0) {
        std::cout << matvec_name << " Matvec test passed" << std::endl;
    }
}
