include(AddGoogleTest)
include_directories(${CMAKE_SOURCE_DIR}/src)


set(LIBS FFT_mvec cufft cublas nccl cutensor OpenMP::OpenMP_CXX OpenMP::OpenMP_C ${CUDA_LIBRARIES} HighFive)


# check if ${NUM_PROCS} is defined. if not set it to 2
if(NOT DEFINED NUM_PROCS)
    set(NUM_PROCS 2)
    message(STATUS "NUM_PROCS not defined. Setting to 2")
endif()

add_gtest(UtilsTest test_Utils.cpp "${LIBS}" 1)
add_gtest(UtilKernelsTest test_UtilKernels.cpp "${LIBS}" 1)
add_gtest(CommTest test_Comm.cpp "${LIBS}" ${NUM_PROCS})
add_gtest(VectorTest test_Vector.cpp "${LIBS}" ${NUM_PROCS})
add_gtest(MatrixTest test_Matrix.cpp "${LIBS}" ${NUM_PROCS})
add_gtest(MatvecTest test_Matvec.cpp "${LIBS}" ${NUM_PROCS})
