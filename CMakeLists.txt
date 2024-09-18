
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fft_matvec LANGUAGES CXX CUDA C)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDA REQUIRED)
find_package(HDF5 REQUIRED)





if (MPI_C_INCLUDE_DIRS STREQUAL "" OR NOT EXISTS MPI_C_INCLUDE_DIRS)
    if (DEFINED MPI_INCLUDE AND NOT MPI_INCLUDE STREQUAL "")
        set (MPI_C_INCLUDE_DIRS ${MPI_INCLUDE})
        set (MPI_CXX_INCLUDE_DIRS ${MPI_INCLUDE})
    else()
        message(FATAL_ERROR "MPI_INCLUDE is not defined")
    endif()
endif()

if (MPI_C_LIBRARIES STREQUAL "" OR NOT EXISTS MPI_C_LIBRARIES)
    if (DEFINED MPI_LIBRARIES AND NOT MPI_LIBRARIES STREQUAL "")
        set (MPI_C_LIBRARIES ${MPI_LIBRARIES})
        set (MPI_CXX_LIBRARIES ${MPI_LIBRARIES})
    else()
        message(FATAL_ERROR "MPI_LIBRARIES is not defined")
    endif()
endif()

if (HDF5_INCLUDE_DIRS STREQUAL "" OR NOT EXISTS HDF5_INCLUDE_DIRS)
message("HDF5_INCLUDE_DIRS is not defined")
    if (DEFINED HDF5_INCLUDE AND NOT HDF5_INCLUDE STREQUAL "")
        set (HDF5_INCLUDE_DIRS ${HDF5_INCLUDE})
    else()
        message(FATAL_ERROR "HDF5_INCLUDE is not defined")
    endif()
endif()


if (HDF5_LIBRARIES STREQUAL "" OR NOT EXISTS HDF5_LIBRARIES)
    if (DEFINED HDF5_LIBRARIES AND NOT HDF5_LIBRARIES STREQUAL "")
        set (HDF5_LIBRARIES ${HDF5_LIBRARIES})
    else()
        message(FATAL_ERROR "HDF5_LIBRARIES is not defined")
    endif()
endif()








INCLUDE_DIRECTORIES(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${NCCL_INCLUDE_DIRS} ${MPI_C_INCLUDE_DIRS} ${MPI_CXX_INCLUDE_DIRS} ${CUTENSOR_ROOT}/include HighFive/include ${HDF5_INCLUDE_DIRS})
LINK_DIRECTORIES(${NCCL_LIBRARIES} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUTENSOR_ROOT}/lib/12/ ${HDF5_LIBRARIES})


add_subdirectory(HighFive)
add_subdirectory(src)


