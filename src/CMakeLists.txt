# --- File Categorization ---

# 1. Pure C++ Headers (no CUDA/HIP specific syntax, or if any, it's guarded by preprocessor)
set(FFT_MVEC_PURE_CPP_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/cmdparser.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/table.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/profiler.hpp"
)

# 2. Pure C++ Source Files
set(FFT_MVEC_PURE_CPP_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/profiler.cpp"
    # Add cmdparser.cpp and table.cpp here if they exist and are pure C++
    # If cmdparser.hpp and table.hpp are header-only, this list is fine.
)

# 3. CUDA-entangled Headers (contain CUDA types/functions, will be hipified for HIP build)
set(FFT_MVEC_CUDA_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/shared.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Comm.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Matrix.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Vector.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tester.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/util_kernels.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/error_checkers.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/comm_error_checkers.h"
)

# 4. CUDA-entangled Source Files (contain CUDA types/functions or are .cu files, will be hipified for HIP build)
# Note: .cpp files with CUDA code will be treated like .cu files for hipification.
set(FFT_MVEC_CUDA_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/util_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/Comm.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Matrix.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/Vector.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tester.cpp"
)

set(ORIGINAL_MAIN_CPP "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp")

# --- Library Definition: FFT_mvec ---
if(BUILD_WITH_HIP)
    set(ORIGINAL_CUDA_SRC_DIR_FOR_FFT_MVEC ${CMAKE_CURRENT_SOURCE_DIR})
    set(HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC ${CMAKE_CURRENT_BINARY_DIR}/hipified)
    set(FILES_TO_HIPIFY ${FFT_MVEC_CUDA_HEADERS} ${FFT_MVEC_CUDA_SOURCES})

    set(CURRENT_HIPIFY_EXTRA_INCLUDES "")
    # Add CUDA toolkit and cuTensor includes gathered from top-level CMakeLists.txt

    list(APPEND CURRENT_HIPIFY_EXTRA_INCLUDES ${HIPIFY_CLANG_CUDA_INCLUDE_DIRS})
    # Add other includes that hipify-clang might need to parse your project files correctly
    list(APPEND CURRENT_HIPIFY_EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}") # For local includes within src

    list(APPEND CURRENT_HIPIFY_EXTRA_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/../HighFive/include")

    list(APPEND DEFINES_FOR_HIPIFY "CUTENSOR_AVAILABLE=0")
    hipify_directory(
        INPUT_FILES_LIST "${FILES_TO_HIPIFY}"
        BASE_CUDA_DIR "${ORIGINAL_CUDA_SRC_DIR_FOR_FFT_MVEC}"
        OUTPUT_HIP_DIR "${HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC}"
        GENERATED_HIP_SOURCES_VAR "FFT_MVEC_LIBRARY_HIP_SOURCES"
        GENERATED_HIP_HEADERS_VAR "FFT_MVEC_LIBRARY_HIP_HEADERS"
        EXTRA_HIPIFY_INCLUDE_PATHS "${CURRENT_HIPIFY_EXTRA_INCLUDES}" # Pass the combined list
        HIPIFY_CLANG_DEFINES "${DEFINES_FOR_HIPIFY}"
    )

    hipify_directory(
        INPUT_FILES_LIST "${ORIGINAL_MAIN_CPP}"
        BASE_CUDA_DIR "${ORIGINAL_CUDA_SRC_DIR_FOR_FFT_MVEC}" # main.cpp is directly in CMAKE_CURRENT_SOURCE_DIR
        OUTPUT_HIP_DIR "${HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC}" # Output to the SAME common directory
        GENERATED_HIP_SOURCES_VAR "MAIN_HIP_SOURCE_LIST"
        GENERATED_HIP_HEADERS_VAR "MAIN_HIP_HEADERS" # Likely empty
        EXTRA_HIPIFY_INCLUDE_PATHS "${CURRENT_HIPIFY_EXTRA_INCLUDES}" # Pass the combined list
        HIPIFY_CLANG_DEFINES "${DEFINES_FOR_HIPIFY}"
    )
    if(NOT MAIN_HIP_SOURCE_LIST STREQUAL "")
        list(GET MAIN_HIP_SOURCE_LIST 0 HIP_MAIN_CPP_FILE_FOR_EXEC)
    else()
        message(FATAL_ERROR "Hipified main.cpp was not generated!")
    endif()

    foreach(hip_source_file IN LISTS FFT_MVEC_LIBRARY_HIP_SOURCES)
        if("${hip_source_file}" MATCHES "\\.cpp$")
            message(STATUS "Setting language of ${hip_source_file} to HIP")
            set_source_files_properties("${hip_source_file}" PROPERTIES LANGUAGE HIP)
        endif()
    endforeach()
    # Also do this for the main executable's source file
    if(HIP_MAIN_CPP_FILE_FOR_EXEC AND "${HIP_MAIN_CPP_FILE_FOR_EXEC}" MATCHES "\\.cpp$")
         set_source_files_properties("${HIP_MAIN_CPP_FILE_FOR_EXEC}" PROPERTIES LANGUAGE HIP)
    endif()

    add_library(FFT_mvec STATIC
        ${FFT_MVEC_LIBRARY_HIP_SOURCES} # Hipified CUDA-entangled sources
        ${FFT_MVEC_LIBRARY_HIP_HEADERS} # Hipified CUDA-entangled headers
        ${FFT_MVEC_PURE_CPP_SOURCES} # Pure C++ sources
        ${FFT_MVEC_PURE_CPP_HEADERS} # Pure C++ headers
    )

    if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "")
        set_target_properties(FFT_mvec PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
        message(STATUS "FFT_mvec: Setting HIP_ARCHITECTURES to ${CMAKE_HIP_ARCHITECTURES}")
    else()
        message(WARNING "FFT_mvec: CMAKE_HIP_ARCHITECTURES is not defined or empty. Library might not be compiled for specific HIP architectures.")
    endif()


    target_include_directories(FFT_mvec PUBLIC
        "${HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC}" # For hipified headers
        "${CMAKE_CURRENT_SOURCE_DIR}" # For pure C++ headers from source dir
        "${HIP_INCLUDE_DIRS}/hipblas/"
        "${HIP_INCLUDE_DIRS}/hipfft/"
        "${HIP_INCLUDE_DIRS}/rccl/"
    )

    target_compile_definitions(FFT_mvec PUBLIC CUTENSOR_AVAILABLE=0)

    set(GPU_SPECIFIC_LIBS
        hipblas hipfft rccl # Ensure these are found by top-level CMake 
        ${HIP_LIBRARIES}
    )

    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fopenmp>) # For host OpenMP in .hip.cpp

    # # add target compile options to explicitly specify --offload-arch=arch0 --offload-arch=arch1... for each GPU architecture in CMAKE_HIP_ARCHITECTURES
    # # split CMAKE_HIP_ARCHITECTURES by ; and add a separate --offload-arch= for each
    # string(REPLACE ";" " --offload-arch=" CMAKE_HIP_ARCHITECTURES_LIST "${CMAKE_HIP_ARCHITECTURES}")
    # foreach(arch IN LISTS CMAKE_HIP_ARCHITECTURES_LIST)
    #     target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=${arch}>)
    # endforeach()

    set(MAIN_EXECUTABLE_SOURCE ${HIP_MAIN_CPP_FILE_FOR_EXEC})

else() # Building with CUDA
    add_library(FFT_mvec STATIC
        ${FFT_MVEC_CUDA_SOURCES} # Original CUDA-entangled sources
        ${FFT_MVEC_CUDA_HEADERS} # Original CUDA-entangled headers
        ${FFT_MVEC_PURE_CPP_SOURCES} # Pure C++ sources
        ${FFT_MVEC_PURE_CPP_HEADERS} # Pure C++ headers
    )

    target_include_directories(FFT_mvec PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}" # For all headers from the source directory
        ${NCCL_INCLUDE_DIRS}

    )

    if(USE_CUTENSOR)
        target_compile_definitions(FFT_mvec PUBLIC CUTENSOR_AVAILABLE=1)
        target_include_directories(FFT_mvec PUBLIC ${CUTENSOR_ROOT}/include)
        target_link_directories(FFT_mvec PUBLIC ${CUTENSOR_ROOT}/lib/12)
    else()
        target_compile_definitions(FFT_mvec PUBLIC CUTENSOR_AVAILABLE=0)
    endif()

    target_link_directories(FFT_mvec PUBLIC ${NCCL_LIBRARIES} ${CUDA_LIBRARIES} ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
    if(USE_CUTENSOR)
        set(GPU_SPECIFIC_LIBS
            cufft cublas nccl cutensor
            ${CUDAToolkit_LIBRARIES}
        )
    else()

        set(GPU_SPECIFIC_LIBS
            cufft cublas nccl
            ${CUDAToolkit_LIBRARIES}
        )
    endif()

    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)

    set(MAIN_EXECUTABLE_SOURCE ${ORIGINAL_MAIN_CPP})
endif()

# --- Executable: fft_matvec ---
add_executable(fft_matvec "${MAIN_EXECUTABLE_SOURCE}")

if(BUILD_WITH_HIP)
    if(DEFINED CMAKE_HIP_ARCHITECTURES AND NOT CMAKE_HIP_ARCHITECTURES STREQUAL "")
        set_target_properties(fft_matvec PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
        message(STATUS "fft_matvec: Setting HIP_ARCHITECTURES to ${CMAKE_HIP_ARCHITECTURES}")
    endif()
    target_include_directories(fft_matvec PRIVATE
        "${HIP_GENERATED_OUTPUT_DIR}" # In case hipified main.cpp includes other hipified headers from the lib
    )
endif()


# The hipified main.cpp will find hipified library headers (like shared.hpp)
# because FFT_mvec publicly includes HIP_GENERATED_OUTPUT_DIR.
# No extra target_include_directories for fft_matvec specifically for HIP_GENERATED_OUTPUT_DIR should be needed here
# unless main.cpp generates its own unique headers into that directory (unlikely for main.cpp).

set(COMMON_LINK_LIBS
    OpenMP::OpenMP_CXX
    HighFive::HighFive
)
target_link_libraries(fft_matvec PUBLIC
    FFT_mvec
    ${COMMON_LINK_LIBS}
    ${GPU_SPECIFIC_LIBS} #are transitively linked via FFT_mvec
)

target_compile_options(fft_matvec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
if(BUILD_WITH_HIP)
    target_compile_options(fft_matvec PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fopenmp>)
    target_compile_definitions(fft_matvec PRIVATE
        CUTENSOR_AVAILABLE=0
        # Add other necessary defines for HIP compilation of main.cpp
    )
else() # CUDA
    target_compile_options(fft_matvec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif()

if(BUILD_WITH_HIP AND TARGET HipifyAll)
    add_dependencies(FFT_mvec HipifyAll)
    add_dependencies(fft_matvec HipifyAll)
endif()
