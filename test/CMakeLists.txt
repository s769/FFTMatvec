include(AddGoogleTest)
include_directories(${CMAKE_SOURCE_DIR}/src)


set(LIBS FFT_mvec cufft cublas nccl cutensor OpenMP::OpenMP_CXX OpenMP::OpenMP_C ${CUDA_LIBRARIES} HighFive)

add_gtest(UtilsTest test_utils.cpp "${LIBS}" 2)
