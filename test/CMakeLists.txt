include(AddGoogleTest)
include_directories(${CMAKE_SOURCE_DIR}/src)
LINK_DIRECTORIES(${NCCL_LIBRARIES} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} cutranspose/build)

add_gtest(CommTest Comm_test.cpp FFT_mvec cufft cublas cuTranspose nccl MPI::MPI_CXX MPI::MPI_C OpenMP::OpenMP_CXX OpenMP::OpenMP_C)
