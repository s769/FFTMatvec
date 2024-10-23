
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(fft_matvec LANGUAGES CXX CUDA C)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules")
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()


find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED)






include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${NCCL_INCLUDE_DIRS} ${CUTENSOR_ROOT}/include HighFive/include)
link_directories(${NCCL_LIBRARIES} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES}  ${CUTENSOR_ROOT}/lib/12/)


add_subdirectory(HighFive)

# if H5_HAVE_PARALLEL is not true, error out
if (NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "HDF5 is not built with parallel support. HDF5_HAVE_PARALLEL: ${HDF5_IS_PARALLEL}")
endif()

add_subdirectory(src)


option(ENABLE_TESTING "Enable testing" ON)

if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test)
endif()
