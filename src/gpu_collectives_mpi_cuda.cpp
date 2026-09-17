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

using MpiQueryFn = int (*)();

MpiQueryFn dlsym_query(const char* name) {
    return reinterpret_cast<MpiQueryFn>(dlsym(RTLD_DEFAULT, name));
}

bool query_mpi_gpu_support(const char* name) {
    MpiQueryFn fn = dlsym_query(name);
    return fn && fn() != 0;
}

// Cray MPICH GPU Transport Layer (GTL). On Frontier AMD, device MPI requires
// linking -lmpi_gtl_hsa and MPICH_GPU_SUPPORT_ENABLED=1. SCALE CUDA pointers
// are HIP device memory, so HSA GTL is the correct backend.
bool mpi_gtl_present() {
    // RTLD_NOLOAD: succeed only if already mapped (e.g. linked into the binary).
    if (void* h = dlopen("libmpi_gtl_hsa.so", RTLD_NOW | RTLD_NOLOAD)) {
        dlclose(h);
        return true;
    }
    if (void* h = dlopen("libmpi_gtl_cuda.so", RTLD_NOW | RTLD_NOLOAD)) {
        dlclose(h);
        return true;
    }
    // Linked but soname differs: look for a GTL export.
    if (dlsym(RTLD_DEFAULT, "MPIX_GPU_query_support") || dlsym(RTLD_DEFAULT, "mpix_gtl_map_gpu_buf")) {
        return true;
    }
    return false;
}

bool mpi_env_gpu_support_enabled() {
    return env_truthy(std::getenv("MPICH_GPU_SUPPORT_ENABLED")) ||
           env_truthy(std::getenv("MPICH_GPU_SUPPORT_ENABLED_CUDA")) ||
           env_truthy(std::getenv("MPICH_GPU_SUPPORT_ENABLED_HIP"));
}

// True when MPI accepts the application's device pointers directly (native CUDA
// GTL, or user force). SCALE cudaMalloc pointers are NOT recognized by Cray's
// HSA GTL even though hipMalloc pointers are — see hip-scratch path below.
bool mpi_accepts_app_device_pointers() {
    if (env_truthy(std::getenv("FFT_MVEC_FORCE_HOST_MPI"))) {
        return false;
    }
    if (env_truthy(std::getenv("FFT_MVEC_ASSUME_GPU_AWARE_MPI"))) {
        return true;
    }
    if (query_mpi_gpu_support("MPIX_Query_cuda_support")) {
        return true;
    }
    if (env_truthy(std::getenv("OMPI_MCA_opal_cuda_support"))) {
        return true;
    }
    return false;
}

struct HipFns {
    using MallocFn = int (*)(void**, size_t);
    using FreeFn = int (*)(void*);
    MallocFn malloc_fn = nullptr;
    FreeFn free_fn = nullptr;
    bool ok = false;
};

HipFns& hip_fns() {
    static HipFns fns = []() {
        HipFns out{};
        // Prefer the system ROCm HIP runtime. SCALE ships its own libamdhip64;
        // Cray GTL IPC attach fails on allocations from that copy.
        const char* rocm = std::getenv("ROCM_PATH");
        std::string candidates[6];
        int n = 0;
        if (rocm && *rocm) {
            candidates[n++] = std::string(rocm) + "/lib/libamdhip64.so.7";
            candidates[n++] = std::string(rocm) + "/lib/libamdhip64.so";
            candidates[n++] = std::string(rocm) + "/lib64/libamdhip64.so.7";
        }
        candidates[n++] = "/opt/rocm-7.2.0/lib/libamdhip64.so.7";
        candidates[n++] = "/opt/rocm/lib/libamdhip64.so.7";
        candidates[n++] = "libamdhip64.so.7";
        void* lib = nullptr;
        for (int i = 0; i < n && !lib; ++i) {
            if (candidates[i].empty()) {
                continue;
            }
            lib = dlopen(candidates[i].c_str(), RTLD_NOW | RTLD_GLOBAL);
        }
        if (!lib) {
            lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
        }
        if (!lib) {
            return out;
        }
        out.malloc_fn = reinterpret_cast<HipFns::MallocFn>(dlsym(lib, "hipMalloc"));
        out.free_fn = reinterpret_cast<HipFns::FreeFn>(dlsym(lib, "hipFree"));
        out.ok = (out.malloc_fn != nullptr && out.free_fn != nullptr);
        return out;
    }();
    return fns;
}

// Frontier SCALE path: Cray GTL works on native hipMalloc, and small D2D+MPI
// probes succeed, but realistic buffer sizes fail GTL IPC when RedSCALE's HIP
 // runtime is also loaded. Keep this path opt-in only.
bool mpi_use_hip_scratch() {
    if (env_truthy(std::getenv("FFT_MVEC_FORCE_HOST_MPI"))) {
        return false;
    }
    if (!env_truthy(std::getenv("FFT_MVEC_FORCE_HIP_SCRATCH_MPI"))) {
        return false;
    }
    return hip_fns().ok && (mpi_env_gpu_support_enabled() || mpi_gtl_present() ||
                            query_mpi_gpu_support("MPIX_Query_hip_support"));
}

bool mpi_is_gpu_aware_cuda() {
    // True only when app device pointers can be passed to MPI directly.
    // HIP-scratch is a separate opt-in path and does not count as "gpu-aware"
    // for this predicate's historical meaning.
    return mpi_accepts_app_device_pointers();
}

void warn_hip_scratch_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        std::fprintf(stderr,
                     "FFTMatvec: using HIP-scratch GPU MPI staging "
                     "(SCALE/cudaMalloc pointers are not Cray-GTL-visible; "
                     "collectives copy via hipMalloc then device MPI). "
                     "Set FFT_MVEC_ASSUME_GPU_AWARE_MPI=1 to force direct device MPI, "
                     "or FFT_MVEC_FORCE_HOST_MPI=1 for host staging.\n");
    });
}

void warn_host_staging_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        std::fprintf(stderr,
                     "FFTMatvec: using host-staged MPI collectives for SCALE device buffers. "
                     "Cray GTL does not accept SCALE/cudaMalloc pointers (native hipMalloc works). "
                     "Multi-proc is supported via host staging. "
                     "Optional: FFT_MVEC_FORCE_HIP_SCRATCH_MPI=1 (experimental), "
                     "FFT_MVEC_ASSUME_GPU_AWARE_MPI=1 (direct, usually segfaults on SCALE).\n");
    });
}

struct HipScratch {
    void* ptr = nullptr;
    size_t nbytes = 0;
    HipScratch(size_t n) : nbytes(n) {
        auto& h = hip_fns();
        if (!h.ok || h.malloc_fn(&ptr, n) != 0 || !ptr) {
            throw std::runtime_error("FFTMatvec: hipMalloc failed for MPI scratch staging");
        }
    }
    ~HipScratch() {
        if (ptr) {
            (void)hip_fns().free_fn(ptr);
        }
    }
    HipScratch(const HipScratch&) = delete;
    HipScratch& operator=(const HipScratch&) = delete;
};

void d2d_copy(void* dst, const void* src, size_t nbytes, cudaStream_t stream) {
    if (nbytes == 0 || dst == src) {
        return;
    }
    gpuErrchk(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    sync_stream(stream);
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
    // Single-rank: no communication (avoids host-staging DtoH/HtoD on 1-GPU runs).
    if (ctx.nranks == 1) {
        if (sendbuf != recvbuf) {
            sync_stream(stream);
            gpuErrchk(cudaMemcpyAsync(recvbuf, sendbuf, count * element_bytes(dtype), cudaMemcpyDeviceToDevice,
                                      stream));
        }
        return;
    }
    sync_stream(stream);
    const size_t nbytes = count * element_bytes(dtype);

    if (mpi_accepts_app_device_pointers()) {
        check_mpi(MPI_Bcast(const_cast<void*>((ctx.rank == root) ? sendbuf : recvbuf), mpi_count, mpi_dtype(dtype),
                            root, ctx.comm),
                  "MPI_Bcast");
        return;
    }

    if (mpi_use_hip_scratch()) {
        warn_hip_scratch_once();
        HipScratch scratch(nbytes);
        if (ctx.rank == root) {
            d2d_copy(scratch.ptr, sendbuf, nbytes, stream);
        }
        check_mpi(MPI_Bcast(scratch.ptr, mpi_count, mpi_dtype(dtype), root, ctx.comm), "MPI_Bcast");
        d2d_copy(recvbuf, scratch.ptr, nbytes, stream);
        return;
    }

    warn_host_staging_once();
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
    if (ctx.nranks == 1) {
        if (sendbuf != recvbuf) {
            sync_stream(stream);
            gpuErrchk(cudaMemcpyAsync(recvbuf, sendbuf, count * element_bytes(dtype), cudaMemcpyDeviceToDevice,
                                      stream));
        }
        return;
    }
    sync_stream(stream);
    const size_t nbytes = count * element_bytes(dtype);

    auto mpi_reduce_buf = [&](const void* sbuf, void* rbuf) {
        if (sbuf == rbuf) {
            if (ctx.rank == root) {
                check_mpi(MPI_Reduce(MPI_IN_PLACE, rbuf, mpi_count, mpi_dtype(dtype), MPI_SUM, root, ctx.comm),
                          "MPI_Reduce");
            } else {
                check_mpi(MPI_Reduce(const_cast<void*>(sbuf), rbuf, mpi_count, mpi_dtype(dtype), MPI_SUM, root,
                                     ctx.comm),
                          "MPI_Reduce");
            }
        } else {
            check_mpi(MPI_Reduce(const_cast<void*>(sbuf), rbuf, mpi_count, mpi_dtype(dtype), MPI_SUM, root, ctx.comm),
                      "MPI_Reduce");
        }
    };

    if (mpi_accepts_app_device_pointers()) {
        mpi_reduce_buf(sendbuf, recvbuf);
        return;
    }

    if (mpi_use_hip_scratch()) {
        warn_hip_scratch_once();
        HipScratch send_scratch(nbytes);
        HipScratch recv_scratch(nbytes);
        d2d_copy(send_scratch.ptr, sendbuf, nbytes, stream);
        if (sendbuf == recvbuf) {
            mpi_reduce_buf(send_scratch.ptr, send_scratch.ptr);
            if (ctx.rank == root) {
                d2d_copy(recvbuf, send_scratch.ptr, nbytes, stream);
            }
        } else {
            mpi_reduce_buf(send_scratch.ptr, recv_scratch.ptr);
            if (ctx.rank == root) {
                d2d_copy(recvbuf, recv_scratch.ptr, nbytes, stream);
            }
        }
        return;
    }

    warn_host_staging_once();
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
    if (ctx.nranks == 1) {
        if (sendbuf != recvbuf) {
            sync_stream(stream);
            gpuErrchk(cudaMemcpyAsync(recvbuf, sendbuf, count * element_bytes(dtype), cudaMemcpyDeviceToDevice,
                                      stream));
        }
        return;
    }
    sync_stream(stream);
    const size_t nbytes = count * element_bytes(dtype);

    auto mpi_allreduce_buf = [&](const void* sbuf, void* rbuf) {
        if (sbuf == rbuf) {
            check_mpi(MPI_Allreduce(MPI_IN_PLACE, rbuf, mpi_count, mpi_dtype(dtype), MPI_SUM, ctx.comm),
                      "MPI_Allreduce");
        } else {
            check_mpi(MPI_Allreduce(const_cast<void*>(sbuf), rbuf, mpi_count, mpi_dtype(dtype), MPI_SUM, ctx.comm),
                      "MPI_Allreduce");
        }
    };

    if (mpi_accepts_app_device_pointers()) {
        mpi_allreduce_buf(sendbuf, recvbuf);
        return;
    }

    if (mpi_use_hip_scratch()) {
        warn_hip_scratch_once();
        HipScratch scratch(nbytes);
        d2d_copy(scratch.ptr, sendbuf, nbytes, stream);
        mpi_allreduce_buf(scratch.ptr, scratch.ptr);
        d2d_copy(recvbuf, scratch.ptr, nbytes, stream);
        return;
    }

    warn_host_staging_once();
    std::vector<unsigned char> send_host(nbytes);
    std::vector<unsigned char> recv_host(nbytes);
    gpuErrchk(cudaMemcpy(send_host.data(), sendbuf, nbytes, cudaMemcpyDeviceToHost));
    check_mpi(MPI_Allreduce(send_host.data(), recv_host.data(), mpi_count, mpi_dtype(dtype), MPI_SUM, ctx.comm),
              "MPI_Allreduce");
    gpuErrchk(cudaMemcpy(recvbuf, recv_host.data(), nbytes, cudaMemcpyHostToDevice));
}

} // namespace GpuCollectives
