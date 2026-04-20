#include "gpu_collectives.hpp"

#include <stdexcept>
#include <string>

namespace GpuCollectives {
namespace {

[[noreturn]] void throw_unavailable(const char* what) {
    throw std::runtime_error(std::string("GPU collectives backend unavailable (") + what + ")");
}

} // namespace

Backend backend() { return Backend::Stub; }
bool available() { return false; }

UniqueId get_unique_id() { throw_unavailable("get_unique_id"); }

Comm create_comm(int /*nranks*/, const UniqueId& /*id*/, int /*rank*/, MPI_Comm /*mpi_comm*/, int /*cuda_device_id*/) {
    throw_unavailable("create_comm");
}

int comm_size(const Comm& /*comm*/) { throw_unavailable("comm_size"); }

int comm_device(const Comm& /*comm*/) { throw_unavailable("comm_device"); }

void broadcast(const void* /*sendbuf*/, void* /*recvbuf*/, size_t /*count*/, DataType /*dtype*/, int /*root*/,
               const Comm& /*comm*/, cudaStream_t /*stream*/) {
    throw_unavailable("broadcast");
}

void reduce(const void* /*sendbuf*/, void* /*recvbuf*/, size_t /*count*/, DataType /*dtype*/, ReduceOp /*op*/,
            int /*root*/, const Comm& /*comm*/, cudaStream_t /*stream*/) {
    throw_unavailable("reduce");
}

void allreduce(const void* /*sendbuf*/, void* /*recvbuf*/, size_t /*count*/, DataType /*dtype*/, ReduceOp /*op*/,
               const Comm& /*comm*/, cudaStream_t /*stream*/) {
    throw_unavailable("allreduce");
}

} // namespace GpuCollectives

