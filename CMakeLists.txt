
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fft_matvec LANGUAGES CXX CUDA C)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
# set(MPIEXEC_EXECUTABLE /usr/bin/mpiexec)
# set(MPI_HOME /usr/lib/x86_64-linux-gnu/openmpi/lib)
# set(MPI_CXX_COMPILER /usr/bin/mpicxx)
# set(MPI_C_COMPILER /usr/bin/mpicc)

# set(CMAKE_C_COMPILER /usr/bin/mpicc)
# set(CMAKE_CXX_COMPILER /usr/bin/mpicxx)

# set(CXXFLAGS  "-fopenmp -O3")
# set(CMAKE_CXX_FLAGS "${CXXFLAGS}")
# set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES /usr/local/cuda/include)

# set(CMAKE_CUDA_ARCHITECTURES 61)

# set(HDF5_PREFER_PARALLEL TRUE)
find_package(MPI REQUIRED)
# include(FindHDF5)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
# find_package(SLEPc REQUIRED)
find_package(NCCL REQUIRED)











INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${NCCL_INCLUDE_DIRS} ${PETSC_INCLUDES} ${MPIC_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS} cutranspose/build)
LINK_DIRECTORIES(${NCCL_LIBRARIES} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} cutranspose/build)



add_subdirectory(src)

