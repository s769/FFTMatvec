file(GLOB_RECURSE INC_FILES shared.hpp table.hpp utils.hpp matvec.hpp profiler.hpp Comm.hpp Matrix.hpp Vector.hpp)
file(GLOB_RECURSE SRC_FILES utils.cu matvec.cpp profiler.cpp Comm.cpp Matrix.cpp Vector.cpp)


set(SOURCE_FILES ${INC_FILES} ${SRC_FILES})

add_library(FFT_mvec ${SOURCE_FILES})

add_executable(fft_matvec main.cpp)

# target_compile_features(fft_matvec PUBLIC cxx_std_23)

set_target_properties( fft_matvec
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
# target_compile_options(fft_matvec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)
target_link_options(fft_matvec PUBLIC ${OpenMP_CXX_FLAGS})

target_link_libraries(fft_matvec FFT_mvec cufft cublas cuTranspose nccl MPI::MPI_CXX MPI::MPI_C OpenMP::OpenMP_CXX OpenMP::OpenMP_C)


