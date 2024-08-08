/**
 * @file Comm.hpp
 * @brief Header file for the Comm class.
 */

#ifndef __COMM_HPP__
#define __COMM_HPP__

#include "shared.hpp"
#include "utils.cuh"

/**
 * @class Comm
 * @brief Class representing communication operations.
 */
class Comm {
private:
    MPI_Comm global_comm; /**< Global MPI communicator */
    MPI_Comm row_comm; /**< MPI communicator for row communication */
    MPI_Comm col_comm; /**< MPI communicator for column communication */
    int row_color; /**< Color of the current process in the row communicator */
    int col_color; /**< Color of the current process in the column communicator */
    int world_rank; /**< Rank of the current process in the global communicator */
    int world_size; /**< Total number of processes in the global communicator */
    int device; /**< GPU device ID */
    int proc_rows; /**< Number of process rows */
    int proc_cols; /**< Number of process columns */
    ncclComm_t gpu_row_comm; /**< NCCL communicator for GPU row communication */
    ncclComm_t gpu_col_comm; /**< NCCL communicator for GPU column communication */
    cudaStream_t s; /**< CUDA stream */
    cublasHandle_t cublasHandle; /**< cuBLAS handle */

public:
    /**
     * @brief Constructor for the Comm class.
     * @param comm The MPI communicator.
     * @param proc_rows Number of process rows.
     * @param proc_cols Number of process columns.
     */
    Comm(MPI_Comm comm, int proc_rows, int proc_cols);

    /**
     * @brief Copy constructor for the Comm class.
     * @param comm The Comm object to be copied.
     */
    Comm(Comm& comm)
        : global_comm(comm.global_comm)
        , proc_rows(comm.proc_rows)
        , proc_cols(comm.proc_cols)
        , world_rank(comm.world_rank)
        , world_size(comm.world_size)
        , row_color(comm.row_color)
        , col_color(comm.col_color)
        , device(comm.device)
        , row_comm(comm.row_comm)
        , col_comm(comm.col_comm)
        , gpu_row_comm(comm.gpu_row_comm)
        , gpu_col_comm(comm.gpu_col_comm)
        , s(comm.s)
        , cublasHandle(comm.cublasHandle)
    {
    } // Copy constructor

    /**
     * @brief Destructor for the Comm class.
     */
    ~Comm();

    /**
     * @brief Get the NCCL communicator for GPU row communication.
     * @return The NCCL communicator for GPU row communication.
     */
    ncclComm_t get_gpu_row_comm() { return gpu_row_comm; }

    /**
     * @brief Get the NCCL communicator for GPU column communication.
     * @return The NCCL communicator for GPU column communication.
     */
    ncclComm_t get_gpu_col_comm() { return gpu_col_comm; }

    /**
     * @brief Get the GPU device ID.
     * @return The GPU device ID.
     */
    int get_device() { return device; }

    /**
     * @brief Get the CUDA stream.
     * @return The CUDA stream.
     */
    cudaStream_t get_stream() { return s; }

    /**
     * @brief Get the MPI communicator for row communication.
     * @return The MPI communicator for row communication.
     */
    MPI_Comm get_row_comm() { return row_comm; }

    /**
     * @brief Get the MPI communicator for column communication.
     * @return The MPI communicator for column communication.
     */
    MPI_Comm get_col_comm() { return col_comm; }

    /**
     * @brief Get the global MPI communicator.
     * @return The global MPI communicator.
     */
    MPI_Comm get_global_comm() { return global_comm; }

    /**
     * @brief Get the color of the current process in the row communicator.
     * @return The color of the current process in the row communicator.
     */
    int get_row_color() { return row_color; }

    /**
     * @brief Get the color of the current process in the column communicator.
     * @return The color of the current process in the column communicator.
     */
    int get_col_color() { return col_color; }

    /**
     * @brief Get the rank of the current process in the global communicator.
     * @return The rank of the current process in the global communicator.
     */
    int get_world_rank() { return world_rank; }

    /**
     * @brief Get the total number of processes in the global communicator.
     * @return The total number of processes in the global communicator.
     */
    int get_world_size() { return world_size; }

    /**
     * @brief Get the number of process rows.
     * @return The number of process rows.
     */
    int get_proc_rows() { return proc_rows; }

    /**
     * @brief Get the number of process columns.
     * @return The number of process columns.
     */
    int get_proc_cols() { return proc_cols; }

    /**
     * @brief Get the cuBLAS handle.
     * @return The cuBLAS handle.
     */
    cublasHandle_t get_cublasHandle() { return cublasHandle; }
};

#endif