#ifndef FFTMATVEC_GPU_COLLECTIVES_HPP
#define FFTMATVEC_GPU_COLLECTIVES_HPP

#include "shared.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace GpuCollectives {

enum class Backend : uint8_t {
    Stub = 0,
    Nccl = 1,
    Rccl = 2,
    MpiCuda = 3,
};

enum class DataType : uint8_t {
    Float32 = 0,
    Float64 = 1,
};

enum class ReduceOp : uint8_t {
    Sum = 0,
};

struct UniqueId {
    static constexpr size_t kBytes = 128;
    std::array<unsigned char, kBytes> bytes{};

    void* data() { return bytes.data(); }
    const void* data() const { return bytes.data(); }
    size_t size() const { return bytes.size(); }
};

struct Comm {
    Backend backend = Backend::Stub;
    std::shared_ptr<void> handle;
};

Backend backend();
bool available();

UniqueId get_unique_id();
// mpi_comm / cuda_device_id are used by the MPI+CUDA staging backend; NCCL/RCCL backends ignore them.
Comm create_comm(int nranks, const UniqueId& id, int rank, MPI_Comm mpi_comm = MPI_COMM_NULL,
                 int cuda_device_id = -1);

int comm_size(const Comm& comm);
int comm_device(const Comm& comm);

void broadcast(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, int root, const Comm& comm,
               cudaStream_t stream);
void reduce(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, ReduceOp op, int root, const Comm& comm,
            cudaStream_t stream);
void allreduce(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, ReduceOp op, const Comm& comm,
               cudaStream_t stream);

} // namespace GpuCollectives

#endif // FFTMATVEC_GPU_COLLECTIVES_HPP
