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

# --- Library Definition: FFT_mvec ---
if(BUILD_WITH_HIP)
    set(ORIGINAL_CUDA_SRC_DIR_FOR_FFT_MVEC ${CMAKE_CURRENT_SOURCE_DIR}) # Base for finding original files
    set(HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC ${CMAKE_CURRENT_BINARY_DIR}/hip_generated_src/FFT_mvec)

    # Combine all CUDA-entangled files to be processed by hipify_directory
    set(FILES_TO_HIPIFY
        ${FFT_MVEC_CUDA_HEADERS}
        ${FFT_MVEC_CUDA_SOURCES}
    )

    # This function (defined in top-level CMakeLists.txt) needs to:
    # - Take each file in FILES_TO_HIPIFY.
    # - Determine its relative path from ORIGINAL_CUDA_SRC_DIR_FOR_FFT_MVEC.
    # - Run hipify-clang.
    # - Place the output in HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC, preserving relative directory structure.
    # - Output lists of generated HIP source files (GENERATED_HIP_SOURCES_VAR)
    #   and generated HIP header files (GENERATED_HIP_HEADERS_VAR).
    hipify_directory(
        INPUT_FILES_LIST "${FILES_TO_HIPIFY}" # Pass the explicit list
        BASE_CUDA_DIR "${ORIGINAL_CUDA_SRC_DIR_FOR_FFT_MVEC}"
        OUTPUT_HIP_DIR "${HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC}"
        GENERATED_HIP_SOURCES_VAR "FFT_MVEC_LIBRARY_HIP_SOURCES" # e.g., hipified .cpp, .hip.cpp
        GENERATED_HIP_HEADERS_VAR "FFT_MVEC_LIBRARY_HIP_HEADERS" # e.g., hipified .hpp, .h
    )

    add_library(FFT_mvec STATIC
        ${FFT_MVEC_LIBRARY_HIP_SOURCES} # Hipified CUDA-entangled sources
        ${FFT_MVEC_LIBRARY_HIP_HEADERS} # Hipified CUDA-entangled headers
        ${FFT_MVEC_PURE_CPP_SOURCES} # Pure C++ sources
        ${FFT_MVEC_PURE_CPP_HEADERS} # Pure C++ headers
    )

    target_include_directories(FFT_mvec PUBLIC
        "${HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC}" # For hipified headers
        "${CMAKE_CURRENT_SOURCE_DIR}" # For pure C++ headers from source dir
    )

    set(GPU_SPECIFIC_LIBS
        roc::rocfft roc::rocblas rccl::rccl # Ensure these are found by top-level CMake
        # Add hip::hipTensor or similar if cutensor is used and mapped
        ${HIP_LIBRARIES}
    )

    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fopenmp>) # For host OpenMP in .hip.cpp

else() # Building with CUDA
    add_library(FFT_mvec STATIC
        ${FFT_MVEC_CUDA_SOURCES} # Original CUDA-entangled sources
        ${FFT_MVEC_CUDA_HEADERS} # Original CUDA-entangled headers
        ${FFT_MVEC_PURE_CPP_SOURCES} # Pure C++ sources
        ${FFT_MVEC_PURE_CPP_HEADERS} # Pure C++ headers
    )

    target_include_directories(FFT_mvec PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}" # For all headers from the source directory
    )

    set(GPU_SPECIFIC_LIBS
        cufft cublas nccl cutensor
        ${CUDAToolkit_LIBRARIES}
    )

    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
    target_compile_options(FFT_mvec PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
endif()

# --- Executable: fft_matvec ---
add_executable(fft_matvec main.cpp) # Assuming main.cpp is pure C++ or uses FFT_mvec

set(COMMON_LINK_LIBS
    OpenMP::OpenMP_CXX
    HighFive
)

target_link_libraries(fft_matvec PUBLIC
    FFT_mvec
    ${COMMON_LINK_LIBS}
    ${GPU_SPECIFIC_LIBS}
)

target_compile_options(fft_matvec PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)

if(BUILD_WITH_HIP AND TARGET HipifyAll)
    add_dependencies(FFT_mvec HipifyAll)
endif()
