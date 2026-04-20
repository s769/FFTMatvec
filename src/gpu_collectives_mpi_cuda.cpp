#include "gpu_collectives.hpp"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace GpuCollectives {
namespace {

struct MpiCudaCtx {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int nranks = 1;
    int device = -1;
};

[[noreturn]] void throw_mpi(int err, const char* what) {
    char msg[MPI_MAX_ERROR_STRING];
    int len = 0;
    MPI_Error_string(err, msg, &len);
    throw std::runtime_error(std::string("MPI error in ") + what + ": " + std::string(msg, static_cast<size_t>(len)));
}

void check_mpi(int err, const char* what) {
    if (err != MPI_SUCCESS) {
        throw_mpi(err, what);
    }
}

size_t element_bytes(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return sizeof(float);
    case DataType::Float64:
        return sizeof(double);
    }
    return sizeof(float);
}

MPI_Datatype mpi_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::Float32:
        return MPI_FLOAT;
    case DataType::Float64:
        return MPI_DOUBLE;
    }
    return MPI_FLOAT;
}

MpiCudaCtx& as_mpi(const Comm& comm) {
    if (comm.backend != Backend::MpiCuda || !comm.handle) {
        throw std::runtime_error("Expected MPI+CUDA collectives backend comm");
    }
    return *static_cast<MpiCudaCtx*>(comm.handle.get());
}

void sync_stream(cudaStream_t stream) {
    gpuErrchk(cudaStreamSynchronize(stream));
}

void check_mpi_count(size_t count) {
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("MPI+CUDA collectives: element count exceeds INT_MAX");
    }
}

bool env_truthy(const char* v) {
    if (!v) {
        return false;
    }
    // Treat empty as false; accept common truthy values.
    if (*v == '\0') {
        return false;
    }
    if (std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0 ||
        std::strcmp(v, "on") == 0 || std::strcmp(v, "ON") == 0 || std::strcmp(v, "yes") == 0 ||
        std::strcmp(v, "YES") == 0) {
        return true;
    }
    return false;
}

bool mpi_is_gpu_aware_cuda() {
    // User overrides.
    if (env_truthy(std::getenv("FFT_MVEC_FORCE_HOST_MPI"))) {
        return false;
    }
    if (env_truthy(std::getenv("FFT_MVEC_ASSUME_GPU_AWARE_MPI"))) {
        return true;
    }

    // Try vendor extensions when present without requiring headers.
    // Cray MPICH / MPICH derivatives often provide: int MPIX_Query_cuda_support(void)
    // OpenMPI sometimes provides macros via mpi-ext.h; we avoid compile-time dependence and use dlsym.
    using QueryFn = int (*)();
    void* sym = dlsym(RTLD_DEFAULT, "MPIX_Query_cuda_support");
    if (sym) {
        auto fn = reinterpret_cast<QueryFn>(sym);
        const int r = fn();
        return (r != 0);
    }

    // Environment hints (best-effort only).
    if (env_truthy(std::getenv("MPICH_GPU_SUPPORT_ENABLED")) || env_truthy(std::getenv("MPICH_GPU_SUPPORT_ENABLED_CUDA"))) {
        return true;
    }
    if (env_truthy(std::getenv("OMPI_MCA_opal_cuda_support"))) {
        return true;
    }

    return false;
}

void warn_host_staging_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        std::fprintf(stderr,
                     "FFTMatvec: MPI does not appear CUDA-GPU-aware; falling back to host-staged collectives. "
                     "Set FFT_MVEC_ASSUME_GPU_AWARE_MPI=1 to force device-pointer MPI, or FFT_MVEC_FORCE_HOST_MPI=1 "
                     "to silence this warning and always stage through host.\n");
    });
}

} // namespace

Backend backend() { return Backend::MpiCuda; }
bool available() { return true; }

UniqueId get_unique_id() {
    UniqueId out{};
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<unsigned int> dist;
    for (size_t i = 0; i < out.bytes.size(); i += sizeof(unsigned int)) {
        unsigned int v = dist(gen);
        const size_t n = std::min(sizeof(unsigned int), out.bytes.size() - i);
        std::memcpy(out.bytes.data() + i, &v, n);
    }
    return out;
}

Comm create_comm(int nranks, const UniqueId& /*id*/, int rank, MPI_Comm mpi_comm, int cuda_device_id) {
    if (mpi_comm == MPI_COMM_NULL) {
        throw std::runtime_error("MPI+CUDA collectives: mpi_comm must not be MPI_COMM_NULL");
    }
    int sz = 0;
    check_mpi(MPI_Comm_size(mpi_comm, &sz), "MPI_Comm_size");
    int rnk = 0;
    check_mpi(MPI_Comm_rank(mpi_comm, &rnk), "MPI_Comm_rank");
    if (sz != nranks) {
        throw std::runtime_error("MPI+CUDA collectives: communicator size does not match nranks");
    }
    if (rnk != rank) {
        throw std::runtime_error("MPI+CUDA collectives: communicator rank does not match rank argument");
    }

    auto* raw = new MpiCudaCtx{};
    check_mpi(MPI_Comm_dup(mpi_comm, &raw->comm), "MPI_Comm_dup");
    check_mpi(MPI_Comm_rank(raw->comm, &raw->rank), "MPI_Comm_rank");
    check_mpi(MPI_Comm_size(raw->comm, &raw->nranks), "MPI_Comm_size");
    raw->device = cuda_device_id;

    Comm out;
    out.backend = Backend::MpiCuda;
    out.handle = std::shared_ptr<void>(raw, [](void* p) {
        auto* ctx = static_cast<MpiCudaCtx*>(p);
        if (ctx && ctx->comm != MPI_COMM_NULL) {
            (void)MPI_Comm_free(&ctx->comm);
        }
        delete ctx;
    });
    return out;
}

int comm_size(const Comm& comm) { return as_mpi(comm).nranks; }

int comm_device(const Comm& comm) { return as_mpi(comm).device; }

void broadcast(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, int root, const Comm& comm,
               cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    check_mpi_count(count);
    const int mpi_count = static_cast<int>(count);
    auto& ctx = as_mpi(comm);
    sync_stream(stream);

    if (mpi_is_gpu_aware_cuda()) {
        // GPU-aware MPI path: call MPI directly on device pointers.
        check_mpi(MPI_Bcast(const_cast<void*>((ctx.rank == root) ? sendbuf : recvbuf), mpi_count, mpi_dtype(dtype), root,
                            ctx.comm),
                  "MPI_Bcast");
        return;
    }

    warn_host_staging_once();
    const size_t nbytes = count * element_bytes(dtype);
    std::vector<unsigned char> host(nbytes);
    if (ctx.rank == root) {
        gpuErrchk(cudaMemcpy(host.data(), sendbuf, nbytes, cudaMemcpyDeviceToHost));
    }
    check_mpi(MPI_Bcast(host.data(), mpi_count, mpi_dtype(dtype), root, ctx.comm), "MPI_Bcast");
    gpuErrchk(cudaMemcpy(recvbuf, host.data(), nbytes, cudaMemcpyHostToDevice));
}

void reduce(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, ReduceOp op, int root, const Comm& comm,
            cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    if (op != ReduceOp::Sum) {
        throw std::runtime_error("MPI+CUDA collectives: only ReduceOp::Sum is supported");
    }
    check_mpi_count(count);
    const int mpi_count = static_cast<int>(count);
    auto& ctx = as_mpi(comm);
    sync_stream(stream);

    if (mpi_is_gpu_aware_cuda()) {
        // GPU-aware MPI path: call MPI directly on device pointers.
        check_mpi(MPI_Reduce(const_cast<void*>(sendbuf), recvbuf, mpi_count, mpi_dtype(dtype), MPI_SUM, root, ctx.comm),
                  "MPI_Reduce");
        return;
    }

    warn_host_staging_once();
    const size_t nbytes = count * element_bytes(dtype);
    std::vector<unsigned char> send_host(nbytes);
    std::vector<unsigned char> recv_host(nbytes);
    gpuErrchk(cudaMemcpy(send_host.data(), sendbuf, nbytes, cudaMemcpyDeviceToHost));
    check_mpi(MPI_Reduce(send_host.data(), recv_host.data(), mpi_count, mpi_dtype(dtype), MPI_SUM, root, ctx.comm),
              "MPI_Reduce");
    if (ctx.rank == root) {
        gpuErrchk(cudaMemcpy(recvbuf, recv_host.data(), nbytes, cudaMemcpyHostToDevice));
    }
}

void allreduce(const void* sendbuf, void* recvbuf, size_t count, DataType dtype, ReduceOp op, const Comm& comm,
               cudaStream_t stream) {
    if (count == 0) {
        return;
    }
    if (op != ReduceOp::Sum) {
        throw std::runtime_error("MPI+CUDA collectives: only ReduceOp::Sum is supported");
    }
    check_mpi_count(count);
    const int mpi_count = static_cast<int>(count);
    auto& ctx = as_mpi(comm);
    sync_stream(stream);

    if (mpi_is_gpu_aware_cuda()) {
        // GPU-aware MPI path: call MPI directly on device pointers.
        check_mpi(MPI_Allreduce(const_cast<void*>(sendbuf), recvbuf, mpi_count, mpi_dtype(dtype), MPI_SUM, ctx.comm),
                  "MPI_Allreduce");
        return;
    }

    warn_host_staging_once();
    const size_t nbytes = count * element_bytes(dtype);
    std::vector<unsigned char> send_host(nbytes);
    std::vector<unsigned char> recv_host(nbytes);
    gpuErrchk(cudaMemcpy(send_host.data(), sendbuf, nbytes, cudaMemcpyDeviceToHost));
    check_mpi(MPI_Allreduce(send_host.data(), recv_host.data(), mpi_count, mpi_dtype(dtype), MPI_SUM, ctx.comm),
              "MPI_Allreduce");
    gpuErrchk(cudaMemcpy(recvbuf, recv_host.data(), nbytes, cudaMemcpyHostToDevice));
}

} // namespace GpuCollectives
