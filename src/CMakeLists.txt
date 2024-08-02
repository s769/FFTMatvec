file(GLOB_RECURSE INC_FILES shared.cuh reduction.cuh multiply.cuh utils.cuh matvec.cuh profiler.h dendro.h)
file(GLOB_RECURSE SRC_FILES reduction.cu multiply.cu utils.cu matvec.cu profiler.cpp)


set(SOURCE_FILES ${INC_FILES} ${SRC_FILES})

add_library(FFT_mvec ${SOURCE_FILES})

add_executable(fft_matvec main.cu)

# target_compile_features(fft_matvec PUBLIC cxx_std_23)

set_target_properties( fft_matvec
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
# target_compile_options(fft_matvec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_options(fft_matvec PUBLIC ${OpenMP_CXX_FLAGS})

target_link_libraries(fft_matvec FFT_mvec cufft cublas cuTranspose nccl OpenMP::OpenMP_CXX)


