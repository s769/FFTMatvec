#include "Comm.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "tester.hpp"
#include <mpi.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(_pyFFTMatvec, m) {
  m.doc() = "Complete Python bindings for FFTMatvec";

  // ==========================================
  // 1. Bind Enums and Configs
  // ==========================================
  py::enum_<Precision>(m, "Precision")
      .value("SINGLE", Precision::SINGLE)
      .value("DOUBLE", Precision::DOUBLE)
      .export_values();

  py::class_<MatvecConfig>(m, "MatvecConfig")
      .def(py::init<>())
      .def_readwrite("transpose", &MatvecConfig::transpose)
      .def_readwrite("full", &MatvecConfig::full)
      .def_readwrite("use_aux_mat", &MatvecConfig::use_aux_mat);

  py::class_<MatvecPrecisionConfig>(m, "MatvecPrecisionConfig")
      .def(py::init<>())
      .def_readwrite("broadcast_and_pad",
                     &MatvecPrecisionConfig::broadcast_and_pad)
      .def_readwrite("fft", &MatvecPrecisionConfig::fft)
      .def_readwrite("sbgemv", &MatvecPrecisionConfig::sbgemv)
      .def_readwrite("ifft", &MatvecPrecisionConfig::ifft)
      .def_readwrite("unpad_and_reduce",
                     &MatvecPrecisionConfig::unpad_and_reduce);

  // ==========================================
  // 2. Bind Comm
  // ==========================================
  py::class_<Comm>(m, "Comm")
      .def(py::init([](int proc_rows, int proc_cols) {
             MPI_Comm comm = MPI_COMM_WORLD;
             return new Comm(comm, proc_rows, proc_cols, 0);
           }),
           py::arg("proc_rows"), py::arg("proc_cols"))
      .def("get_device", &Comm::get_device)
      .def("get_world_rank", &Comm::get_world_rank)
      .def("get_world_size", &Comm::get_world_size)
      .def("get_proc_rows", &Comm::get_proc_rows)
      .def("get_proc_cols", &Comm::get_proc_cols)
      .def("get_row_color", &Comm::get_row_color)
      .def("get_col_color", &Comm::get_col_color)
      .def("has_external_stream", &Comm::has_external_stream)
      // Hardware / low-level handles cast to uintptr_t
      .def("get_gpu_row_comm",
           [](Comm &c) {
             return reinterpret_cast<uintptr_t>(c.get_gpu_row_comm());
           })
      .def("get_gpu_col_comm",
           [](Comm &c) {
             return reinterpret_cast<uintptr_t>(c.get_gpu_col_comm());
           })
      .def("get_stream",
           [](Comm &c) { return reinterpret_cast<uintptr_t>(c.get_stream()); })
      .def("get_cublasHandle",
           [](Comm &c) {
             return reinterpret_cast<uintptr_t>(c.get_cublasHandle());
           })
      .def("get_event",
           [](Comm &c) { return reinterpret_cast<uintptr_t>(c.get_event()); })
      // MPI Communicators cast to Fortran handles for mpi4py compatibility
      .def("get_global_comm",
           [](Comm &c) { return MPI_Comm_c2f(c.get_global_comm()); })
      .def("get_row_comm",
           [](Comm &c) { return MPI_Comm_c2f(c.get_row_comm()); })
      .def("get_col_comm",
           [](Comm &c) { return MPI_Comm_c2f(c.get_col_comm()); });

  // ==========================================
  // 3. Bind Vector
  // ==========================================
  py::class_<Vector>(m, "Vector")
      .def(py::init<Comm &, unsigned int, unsigned int, std::string, bool,
                    bool>(),
           py::arg("comm"), py::arg("blocks"), py::arg("block_size"),
           py::arg("row_or_col"), py::arg("global_sizes") = false,
           py::arg("SOTI_ordering") = true)
      .def(py::init<Vector &, bool>(), py::arg("vec"), py::arg("deep_copy"))
      .def(py::init<Vector &>(), py::arg("vec"))

      // Initializations
      .def("init_vec", &Vector::init_vec)
      .def("init_vec_ones", &Vector::init_vec_ones)
      .def("init_vec_zeros", &Vector::init_vec_zeros)
      .def("init_vec_doubles", &Vector::init_vec_doubles)
      .def("init_vec_consecutive", &Vector::init_vec_consecutive)
      .def("init_vec_from_file", &Vector::init_vec_from_file,
           py::arg("filename"), py::arg("checksum") = 0, py::arg("QoI") = false)

      // Mathematical Operators & In-place functions
      .def("scale", &Vector::scale, py::arg("alpha"))
      .def("wscale", &Vector::wscale, py::arg("alpha"))
      .def("axpy", &Vector::axpy, py::arg("alpha"), py::arg("x"))
      .def("waxpy", &Vector::waxpy, py::arg("alpha"), py::arg("x"))
      .def("axpby", &Vector::axpby, py::arg("alpha"), py::arg("beta"),
           py::arg("x"))
      .def("waxpby", &Vector::waxpby, py::arg("alpha"), py::arg("beta"),
           py::arg("x"))
      .def("dot", &Vector::dot, py::arg("x"))
      .def("norm", &Vector::norm, py::arg("order") = 2, py::arg("name") = "")
      .def("elementwise_multiply", &Vector::elementwise_multiply,
           py::arg("other"))
      .def("elementwise_multiply_inplace",
           &Vector::elementwise_multiply_inplace, py::arg("other"))
      .def("elementwise_divide", &Vector::elementwise_divide, py::arg("other"))
      .def("elementwise_divide_inplace", &Vector::elementwise_divide_inplace,
           py::arg("other"))
      .def("elementwise_inverse", &Vector::elementwise_inverse)
      .def("elementwise_inverse_inplace", &Vector::elementwise_inverse_inplace)

      // Standard Python Magic Methods
      .def("__add__", [](Vector &a, Vector &b) { return a + b; })
      .def("__sub__", [](Vector &a, Vector &b) { return a - b; })
      .def("__mul__", [](Vector &a, double alpha) { return a * alpha; })
      .def("__rmul__", [](Vector &a, double alpha) { return alpha * a; })
      .def("__truediv__", [](Vector &a, double alpha) { return a / alpha; })
      .def("__mul__", [](Vector &a, Vector &b) { return a * b; })
      .def("__truediv__", [](Vector &a, Vector &b) { return a / b; })

      // Utilities
      .def("print", &Vector::print, py::arg("name") = "")
      .def("save", &Vector::save, py::arg("filename"), py::arg("QoI") = false)
      .def("copy", &Vector::copy, py::arg("x"))
      .def("on_grid", &Vector::on_grid)

      // Overloaded Memory/Size Operations
      .def("extend", py::overload_cast<int>(&Vector::extend),
           py::arg("new_block_size"))
      .def("extend", py::overload_cast<Vector &>(&Vector::extend),
           py::arg("out"))
      .def("shrink", py::overload_cast<int>(&Vector::shrink),
           py::arg("new_block_size"))
      .def("shrink", py::overload_cast<Vector &>(&Vector::shrink),
           py::arg("out"))
      .def("resize", py::overload_cast<int>(&Vector::resize),
           py::arg("new_block_size"))
      .def("resize", py::overload_cast<Vector &>(&Vector::resize),
           py::arg("out"))

      // Getters & Setters
      .def("get_num_blocks", &Vector::get_num_blocks)
      .def("get_glob_num_blocks", &Vector::get_glob_num_blocks)
      .def("get_padded_size", &Vector::get_padded_size)
      .def("get_block_size", &Vector::get_block_size)
      .def("get_row_or_col", &Vector::get_row_or_col)
      .def("is_initialized", &Vector::is_initialized)
      .def("is_SOTI_ordered", &Vector::is_SOTI_ordered)
      .def("get_checksum", &Vector::get_checksum)
      .def("set_checksum", &Vector::set_checksum, py::arg("checksum"))
      .def("get_d_vec",
           [](Vector &v) { return reinterpret_cast<uintptr_t>(v.get_d_vec()); })
      .def("set_d_vec", [](Vector &v, uintptr_t ptr) {
        v.set_d_vec(reinterpret_cast<double *>(ptr));
      });

  // ==========================================
  // 4. Bind Matrix
  // ==========================================
  py::class_<Matrix>(m, "Matrix")
      .def(py::init<Comm &, unsigned int, unsigned int, unsigned int, bool,
                    bool, const MatvecPrecisionConfig &>(),
           py::arg("comm"), py::arg("cols"), py::arg("rows"),
           py::arg("block_size"), py::arg("global_sizes") = false,
           py::arg("QoI") = false,
           py::arg("p_config") = MatvecPrecisionConfig())
      .def(py::init<Comm &, std::string, std::string, bool,
                    const MatvecPrecisionConfig &>(),
           py::arg("comm"), py::arg("path"), py::arg("aux_path") = "",
           py::arg("QoI") = false,
           py::arg("p_config") = MatvecPrecisionConfig())

      // Initializations
      .def("init_mat_from_file", &Matrix::init_mat_from_file,
           py::arg("dirname"), py::arg("aux_mat") = false)
      .def("init_mat_ones", &Matrix::init_mat_ones, py::arg("aux_mat") = false)
      .def("init_mat_doubles", &Matrix::init_mat_doubles,
           py::arg("aux_mat") = false)

      // Core Operations
      .def("matvec", &Matrix::matvec, py::arg("x"), py::arg("y"),
           py::arg("use_aux_mat") = false, py::arg("full") = false)
      .def("transpose_matvec", &Matrix::transpose_matvec, py::arg("x"),
           py::arg("y"), py::arg("use_aux_mat") = false,
           py::arg("full") = false)
      .def("get_vec", &Matrix::get_vec, py::arg("input_or_output"))

      // Standard Getters
      .def("get_num_cols", &Matrix::get_num_cols)
      .def("get_num_rows", &Matrix::get_num_rows)
      .def("get_glob_num_cols", &Matrix::get_glob_num_cols)
      .def("get_glob_num_rows", &Matrix::get_glob_num_rows)
      .def("get_padded_size", &Matrix::get_padded_size)
      .def("get_block_size", &Matrix::get_block_size)
      .def("is_initialized", &Matrix::is_initialized)
      .def("has_aux_mat", &Matrix::has_aux_mat)
      .def("is_p2q_mat", &Matrix::is_p2q_mat)
      .def("get_checksum", &Matrix::get_checksum)
      .def("get_precision_config", &Matrix::get_precision_config)

      // Raw GPU Pointers (Doubles & Floats) -> cast to uintptr_t
      .def("get_col_vec_unpad",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_unpad());
           })
      .def("get_col_vec_pad",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_pad());
           })
      .def("get_row_vec_pad",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_pad());
           })
      .def("get_row_vec_unpad",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_unpad());
           })
      .def("get_res_pad",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_res_pad());
           })
      .def("get_col_vec_unpad_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_unpad_F());
           })
      .def("get_col_vec_pad_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_pad_F());
           })
      .def("get_row_vec_pad_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_pad_F());
           })
      .def("get_row_vec_unpad_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_unpad_F());
           })

      // Complex GPU Pointers -> cast to uintptr_t
      .def("get_col_vec_freq",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_freq());
           })
      .def("get_row_vec_freq",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_freq());
           })
      .def("get_col_vec_freq_TOSI",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_freq_TOSI());
           })
      .def("get_row_vec_freq_TOSI",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_freq_TOSI());
           })
      .def("get_mat_freq_TOSI",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_mat_freq_TOSI());
           })
      .def("get_mat_freq_TOSI_aux",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_mat_freq_TOSI_aux());
           })
      .def("get_col_vec_freq_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_freq_F());
           })
      .def("get_row_vec_freq_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_freq_F());
           })
      .def("get_col_vec_freq_TOSI_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_col_vec_freq_TOSI_F());
           })
      .def("get_row_vec_freq_TOSI_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_row_vec_freq_TOSI_F());
           })
      .def("get_mat_freq_TOSI_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_mat_freq_TOSI_F());
           })
      .def("get_mat_freq_TOSI_aux_F",
           [](Matrix &m) {
             return reinterpret_cast<uintptr_t>(m.get_mat_freq_TOSI_aux_F());
           })

      // cuFFT Handles
      .def("get_forward_plan", &Matrix::get_forward_plan)
      .def("get_inverse_plan", &Matrix::get_inverse_plan)
      .def("get_forward_plan_conj", &Matrix::get_forward_plan_conj)
      .def("get_inverse_plan_conj", &Matrix::get_inverse_plan_conj);

  // ==========================================
  // 5. Bind Tester Namespace
  // ==========================================
  // Create a submodule named "Tester" inside the pyFFTMatvec module
  py::module_ tester_m =
      m.def_submodule("Tester", "Testing utilities for FFTMatvec");

  // Bind the check_ones_matvec free function
  tester_m.def("check_ones_matvec", &Tester::check_ones_matvec,
               "Checks the results of a matrix-vector multiplication with a "
               "ones matrix and ones vectors.",
               py::arg("comm"), py::arg("mat"), py::arg("out"), py::arg("conj"),
               py::arg("full"));
}
