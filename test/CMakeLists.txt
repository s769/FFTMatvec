include(AddGoogleTest)
INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${NCCL_INCLUDE_DIRS} ${CUTENSOR_ROOT}/include HighFive/include ${CMAKE_SOURCE_DIR}/src)
LINK_DIRECTORIES(${NCCL_LIBRARIES} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUTENSOR_ROOT}/lib/12/)


add_gtest(UtilsTest test_utils.cpp FFT_mvec cufft cublas nccl cutensor mpi_nvidia OpenMP::OpenMP_CXX OpenMP::OpenMP_C ${CUDA_LIBRARIES} HighFive)