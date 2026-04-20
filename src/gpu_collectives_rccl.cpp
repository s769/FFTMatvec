#include "gpu_collectives.hpp"

#include <rccl.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace GpuCollectives {
namespace {

[[noreturn]] void throw_rccl(ncclResult_t r, const char* what) {
    // RCCL uses NCCL-compatible types but provides its own error string function.
    throw std::runtime_error(std::string("RCCL error in ") + what + ": " + ncclGetErrorString(r));
}

ncclDataType_t to_rccl(DataType t) {
    switch (t) {
    case DataType::Float32:
        return ncclFloat;
    case DataType::Float64:
        return ncclDouble;
    }
    return ncclFloat;
}

ncclRedOp_t to_rccl(ReduceOp op) {
    switch (op) {
    case ReduceOp::Sum:
        return ncclSum;
    }
    return ncclSum;
}

ncclComm_t& as_rccl(const Comm& comm) {
    if (comm.backend != Backend::Rccl || !comm.handle) {
        throw std::runtime_error("Expected RCCL backend comm");
    }
    return *reinterpret_cast<ncclComm_t*>(comm.handle.get());
}

} // namespace

Backend backend() { return Backend::Rccl; }
bool available() { return true; }

UniqueId get_unique_id() {
    ncclUniqueId id{};
    ncclResult_t r = ncclGetUniqueId(&id);
    if (r != ncclSuccess) {
        throw_rccl(r, "ncclGetUniqueId");
    }

    UniqueId out;
    static_assert(sizeof(ncclUniqueId) <= UniqueId::kBytes, "UniqueId storage too small");
    std::memcpy(out.bytes.data(), &id, sizeof(ncclUniqueId));
    return out;
}

Comm create_comm(int nranks, const UniqueId& id, int rank, MPI_Comm /*mpi_comm*/, int /*cuda_device_id*/) {
    ncclUniqueId native{};
    std::memcpy(&native, id.bytes.data(), sizeof(ncclUniqueId));

    auto* c = new ncclComm_t{};
    ncclResult_t r = ncclCommInitRank(c, nranks, native, rank);
    if (r != ncclSuccess) {
        delete c;
        throw_rccl(r, "ncclCommInitRank");
    }

    Comm out;
    out.backend = Backend::Rccl;
    out.handle = std::shared_ptr<void>(c, [](void* p) {
        auto* comm = reinterpret_cast<ncclComm_t*>(p);
        if (comm) {
            (void)ncclCommDestroy(*comm);
            delete comm;
        }
    });
    return out;
}

int comm_size(const Comm& comm) {
    int n = 0;
    ncclResult_t r = ncclCommCount(as_rccl(comm), &n);
    if (r != ncclSuccess) {
        throw_rccl(r, "ncclCommCount");
    }
    return n;
}

int comm_device(const Comm& comm) {
    // RCCL does not always support ncclCommCuDevice the same way; try it and surface errors.
    int dev = -1;
    ncclResult_t r = ncclCommCuDevice(as_rccl(comm), &dev);
    if (r != ncclSuccess) {
        throw_rccl(r, "ncclCommCuDevice");
    }
    return dev;
}

void broadcast(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, int root, const Comm& comm,
               cudaStream_t stream) {
    ncclResult_t r = ncclBroadcast(sendbuf, recvbuf, count, to_rccl(dtype), root, as_rccl(comm), stream);
    if (r != ncclSuccess) {
        throw_rccl(r, "ncclBroadcast");
    }
}

void reduce(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, ReduceOp op, int root, const Comm& comm,
            cudaStream_t stream) {
    ncclResult_t r = ncclReduce(sendbuf, recvbuf, count, to_rccl(dtype), to_rccl(op), root, as_rccl(comm), stream);
    if (r != ncclSuccess) {
        throw_rccl(r, "ncclReduce");
    }
}

void allreduce(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, ReduceOp op, const Comm& comm,
               cudaStream_t stream) {
    ncclResult_t r = ncclAllReduce(sendbuf, recvbuf, count, to_rccl(dtype), to_rccl(op), as_rccl(comm), stream);
    if (r != ncclSuccess) {
        throw_rccl(r, "ncclAllReduce");
    }
}

} // namespace GpuCollectives

