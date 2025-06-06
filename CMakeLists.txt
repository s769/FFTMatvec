cmake_minimum_required(VERSION 3.21 FATAL_ERROR) # Recommended for best HIP support
project(fft_matvec LANGUAGES CXX C) # CUDA/HIP will be enabled conditionally

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake-modules" ${CMAKE_MODULE_PATH}) # Add our module path
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# --- Option for HIP Build ---
option(BUILD_WITH_HIP "Build with HIP support instead of CUDA" OFF)

if(BUILD_WITH_HIP)
    message(STATUS "Selected GPU Platform: HIP")
    enable_language(HIP)
    find_package(HIP REQUIRED) # Finds hipcc, HIP_LIBRARIES, HIP_INCLUDE_DIRS
    message(STATUS "HIP_INCLUDE_DIRS ${HIP_INCLUDE_DIRS}")
    include(HipifyHelpers) # Include our custom module


    # User needs to set AMDGPU_TARGETS, e.g., "gfx900;gfx906;gfx908;gfx1030"
    # You can set it via cmake -DAMDGPU_TARGETS="gfx908" or an environment variable
    # 1. Define AMDGPU_TARGETS as a CACHE variable so the user's -D input is stored and documented.
    #    Provide a sensible default if you wish.
    set(AMDGPU_TARGETS_DEFAULT "gfx90a;gfx942" CACHE STRING "Semicolon-separated list of AMD GPU architectures (e.g., gfx90a;gfx942)")

    # 2. Determine the final value to use for architectures.
    #    Prioritize user-provided AMDGPU_TARGETS if it's different from the default or explicitly set.
    #    This logic ensures that what the user passes via -D AMDGPU_TARGETS=... takes precedence.
    set(FINAL_ARCHITECTURES_TO_SET "${AMDGPU_TARGETS}") # AMDGPU_TARGETS will hold the value from -D or the cached default.

    if(NOT FINAL_ARCHITECTURES_TO_SET)
        message(WARNING "AMDGPU_TARGETS is empty. Please specify target AMD GPU architectures. Defaulting to 'gfx90a'.")
        set(FINAL_ARCHITECTURES_TO_SET "gfx90a;gfx942")
    endif()

    set(CMAKE_HIP_ARCHITECTURES "${FINAL_ARCHITECTURES_TO_SET}" CACHE STRING "AMD GPU architectures for HIP compilation" FORCE)


    set(HIPIFY_CLANG_CUDA_INCLUDE_DIRS "")

    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND AND CUDAToolkit_INCLUDE_DIRS)
        list(APPEND HIPIFY_CLANG_CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
        # Some hipify-clang versions also benefit from knowing the CUDA root
        # set(HIPIFY_CLANG_CUDA_ROOT_ARG --cuda-path=${CUDAToolkit_TOOLKIT_ROOT_DIR}) # We'll pass this to hipify_helpers if needed
    else()
        message(STATUS "CUDAToolkit not found or include_dirs not set. Hipify-clang might have trouble finding standard CUDA headers.")
    endif()

    list(APPEND HIPIFY_CLANG_CUDA_INCLUDE_DIRS ${NCCL_INCLUDE_DIRS})



    # Global target to trigger all hipification (optional, dependencies are usually file-based)
    # add_custom_target(HipifyAll COMMENT "Running all hipify commands")

    # Find HIP equivalent libraries (rocBLAS, rocFFT, RCCL, hipTensor etc.)
    find_package(rocblas REQUIRED) # Provides roc::rocblas target
    find_package(rocfft REQUIRED) # Provides roc::rocfft target
    find_package(rccl REQUIRED) # Provides rccl::rccl target (or similar)
    # find_package(hipTensor) # If used and available

    set(GPU_PLATFORM "HIP")

else() # CUDA Build
    message(STATUS "Selected GPU Platform: CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED) # Ensures CUDA toolkit components are found

    if(NOT DEFINED CUDA_ARCH)
        message(STATUS "CUDA_ARCH not specified. Setting CMAKE_CUDA_ARCHITECTURES to 80;90")
        set(CMAKE_CUDA_ARCHITECTURES "80;90")
    else()
        message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to ${CUDA_ARCH}")
        set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
    endif()
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

    if(NOT "${CUTENSOR_ROOT}" STREQUAL "")
        message(STATUS "Using cuTensor from CUTENSOR_ROOT: ${CUTENSOR_ROOT}")
        set(USE_CUTENSOR ON)
    else()
        message(STATUS "CUTENSOR_ROOT not set. cutranspose will be used for swapaxes.")
        set(USE_CUTENSOR OFF)
    endif()


    # Find CUDA specific libraries
    # find_package(CUDAToolkit REQUIRED COMPONENTS cublas cufft curand nvjpeg ...)
    # Modern CMake uses targets like CUDAToolkit::cublas, CUDAToolkit::cufft
    # For NCCL and cuTensor, you might need custom Find modules or ensure they provide imported targets.
    # Example: find_package(NCCL) -> provides NCCL::NCCL target
    # Example: find_package(cuTensor) -> provides CUDA::cutensor_static target

    # For now, we assume these will be linked directly by name if specific targets aren't found/used,
    # but using imported targets (e.g., CUDAToolkit::cublas) is preferred.
    # The src/CMakeLists.txt will list 'cublas', 'cufft', 'nccl', 'cutensor'.
    # If you have FindNCCL.cmake or FindcuTensor.cmake, ensure they are in CMAKE_MODULE_PATH
    # and that they set up imported targets or standard variables.
    # Let's assume for now the linking in src/CMakeLists.txt will use the raw names,
    # and CMake's default CUDA search paths or your environment helps find them.
    # A more robust setup uses specific find_package calls for each.
    # E.g. for NCCL:
    # find_package(NCCL) # You'd need a FindNCCL.cmake module for this if not standard
    # if(NCCL_FOUND)
    # message(STATUS "Found NCCL: ${NCCL_INCLUDE_DIRS}, ${NCCL_LIBRARIES}")
    # else()
    # message(WARNING "NCCL not found for CUDA build.")
    # endif()
    # Similar for cuTensor
    # find_package(cuTensor)
    # if(cuTensor_FOUND)
    # message(STATUS "Found cuTensor: ${cuTensor_INCLUDE_DIRS}, ${cuTensor_LIBRARIES}")
    # else()
    # message(WARNING "cuTensor not found for CUDA build.")
    # endif()
    set(GPU_PLATFORM "CUDA")

endif()

# --- Common Packages & Settings ---
set(CMAKE_CXX_STANDARD 17) # Assuming C++17, adjust as needed
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(BUILD_WITH_HIP)
    set(CMAKE_HIP_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_HIP_STANDARD_REQUIRED ON)
    set(CMAKE_HIP_EXTENSIONS OFF)
else()
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD}) # NVCC uses this for host compilation part
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
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
find_package(HDF5 REQUIRED) # Request CXX and HL components

# Check for HDF5 parallel support (remains important)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "HDF5 is not built with parallel support. HDF5_CXX_IS_PARALLEL: ${HDF5_CXX_IS_PARALLEL}")
endif()

# HighFive is added as a subdirectory, it should set up its own targets and includes
add_subdirectory(HighFive) # Assuming HighFive/CMakeLists.txt handles its linkage and includes correctly

# --- Global Include Directories (for common libraries like MPI, HDF5) ---
# Prefer target_include_directories for project sources, but global ones are okay for widely used external libs.
include_directories(
    ${MPI_INCLUDE_PATH} # From find_package(MPI)
    ${HDF5_INCLUDE_DIRS} # From find_package(HDF5)
    HighFive/include
    # HighFive should make its includes available via its target (e.g. HighFive::HighFive)
)
# Note: GPU specific includes (CUDA Toolkit, HIP, rocBLAS includes etc.) should be handled by
# find_package results (e.g. ${HIP_INCLUDE_DIRS}, ${CUDAToolkit_INCLUDE_DIRS}) and applied
# via target_include_directories in src/CMakeLists.txt, or by imported targets.
# The old global include_directories for CUDA/NCCL/CUTENSOR has been removed.

# --- Add Source and Test Subdirectories ---
# These subdirectories will use the BUILD_WITH_HIP option.
add_subdirectory(src)

option(ENABLE_TESTING "Enable testing" ON)
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(test) # Assuming test/CMakeLists.txt will also be adapted for CUDA/HIP
endif()

message(STATUS "---------------------------------------------------------")
message(STATUS "Project Name: ${PROJECT_NAME}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "GPU Platform: ${GPU_PLATFORM}") # Set this in src based on BUILD_WITH_HIP
if(BUILD_WITH_HIP)
    message(STATUS "HIP Achitectures: ${CMAKE_HIP_ARCHITECTURES}")
else()
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "MPI CXX Compiler: ${MPI_CXX_COMPILER}")
message(STATUS "HDF5 Version: ${HDF5_VERSION}")
message(STATUS "HDF5 is Parallel: ${HDF5_IS_PARALLEL}")
message(STATUS "---------------------------------------------------------")
