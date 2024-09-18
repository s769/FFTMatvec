file(GLOB_RECURSE INC_FILES shared.hpp table.hpp utils.hpp matvec.hpp profiler.hpp Comm.hpp Matrix.hpp Vector.hpp tester.hpp cmdparser.hpp)
file(GLOB_RECURSE SRC_FILES utils.cu matvec.cpp profiler.cpp Comm.cpp Matrix.cpp Vector.cpp tester.cpp)


set(SOURCE_FILES ${INC_FILES} ${SRC_FILES})
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)


add_library(FFT_mvec ${SOURCE_FILES})


add_executable(fft_matvec main.cpp)


set_target_properties( fft_matvec
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                      )

target_link_libraries(fft_matvec PUBLIC FFT_mvec cufft cublas nccl cutensor mpi_nvidia OpenMP::OpenMP_CXX OpenMP::OpenMP_C ${CUDA_LIBRARIES})

