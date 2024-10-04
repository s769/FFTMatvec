include(AddGoogleTest)
include_directories(${CMAKE_SOURCE_DIR}/src)


set(LIBS FFT_mvec cufft cublas nccl cutensor OpenMP::OpenMP_CXX OpenMP::OpenMP_C ${CUDA_LIBRARIES} HighFive)

add_gtest(UtilsTest test_utils.cpp "${LIBS}" 1)
add_gtest(MatrixTest test_Matrix.cpp "${LIBS}" 2)
add_gtest(CommTest test_Comm.cpp "${LIBS}" 2)
add_gtest(VectorTest test_Vector.cpp "${LIBS}" 2)