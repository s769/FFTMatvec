
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fft_matvec LANGUAGES CXX CUDA C)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED)






INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${NCCL_INCLUDE_DIRS} ${CUTENSOR_ROOT}/include HighFive/include)
LINK_DIRECTORIES(${NCCL_LIBRARIES} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUTENSOR_ROOT}/lib/12/)


add_subdirectory(HighFive)
add_subdirectory(src)


