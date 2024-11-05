file(GLOB_RECURSE INC_FILES shared.hpp table.hpp utils.hpp profiler.hpp Comm.hpp Matrix.hpp Vector.hpp tester.hpp cmdparser.hpp util_kernels.hpp error_checkers.h comm_error_checkers.h)
file(GLOB_RECURSE SRC_FILES utils.cpp util_kernels.cu profiler.cpp Comm.cpp Matrix.cpp Vector.cpp tester.cpp)


set(SOURCE_FILES ${INC_FILES} ${SRC_FILES})
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)


add_library(FFT_mvec STATIC ${SOURCE_FILES})


add_executable(fft_matvec main.cpp)


set(LIBS FFT_mvec cufft cublas nccl cutensor OpenMP::OpenMP_CXX OpenMP::OpenMP_C ${CUDA_LIBRARIES} HighFive)

target_link_libraries(fft_matvec PUBLIC ${LIBS})

