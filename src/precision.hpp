#ifndef __PRECISION_HPP__
#define __PRECISION_HPP__

#include "shared.hpp"

enum class Precision
{
    Single,
    Double
};

struct MatvecPrecisionConfig
{
    Precision broadcast = Precision::Double;
    Precision pad = Precision::Double;
    Precision fft = Precision::Double;
    Precision transpose = Precision::Double;
    Precision sbgemv = Precision::Double;
    Precision ifft = Precision::Double;
    Precision unpad = Precision::Double;
    Precision reduce = Precision::Double;
    Precision setup_fft = Precision::Double;
    Precision setup_swapaxes = Precision::Double;
};

template<typename T_real>
struct TypeTraits;

// Specialization for SINGLE precision
template<>
struct TypeTraits<float> {
    using Real = float;
    using Complex = ComplexF;
    static constexpr cufftType_t cufft_type = CUFFT_R2C; // Single-to-Complex
    static constexpr cudaDataType_t cuda_data_type = CUDA_R_32F;

    // Helper for cuBLAS calls
    template<typename... Args>
    static cublasStatus_t cublasGemv(Args&&... args) {
        return cublasCgemvStridedBatched(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static cublasStatus_t cublasGeam(Args&&... args) {
        return cublasCgeam(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static cublasStatus_t cublasScal(Args&&... args) {
        return cublasCscal(std::forward<Args>(args)...);
    }


};

// Specialization for DOUBLE precision
template<>
struct TypeTraits<double> {
    using Real = double;
    using Complex = ComplexD;
    static constexpr cufftType_t cufft_type = CUFFT_D2Z; // Double-to-Double Complex
    static constexpr cudaDataType_t cuda_data_type = CUDA_R_64F;

    // Helper for cuBLAS calls
    template<typename... Args>
    static cublasStatus_t cublasGemv(Args&&... args) {
        return cublasZgemvStridedBatched(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static cublasStatus_t cublasGeam(Args&&... args) {
        return cublasZgeam(std::forward<Args>(args)...);
    }

    template<typename... Args>
    static cublasStatus_t cublasScal(Args&&... args) {
        return cublasZscal(std::forward<Args>(args)...);
    }
};


#endif // __PRECISION_HPP__