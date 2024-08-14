file(GLOB_RECURSE INC_FILES shared.hpp table.hpp utils.hpp matvec.hpp profiler.hpp Comm.hpp Matrix.hpp Vector.hpp tester.hpp)
file(GLOB_RECURSE SRC_FILES utils.cu matvec.cpp profiler.cpp Comm.cpp Matrix.cpp Vector.cpp tester.cpp)


set(SOURCE_FILES ${INC_FILES} ${SRC_FILES})

add_library(FFT_mvec ${SOURCE_FILES})

add_executable(fft_matvec main.cpp)


set_target_properties( fft_matvec
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )
target_link_options(fft_matvec PUBLIC ${OpenMP_CXX_FLAGS})

target_link_libraries(fft_matvec FFT_mvec cufft cublas nccl cutensor MPI::MPI_CXX MPI::MPI_C OpenMP::OpenMP_CXX OpenMP::OpenMP_C)


