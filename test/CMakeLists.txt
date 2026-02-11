include(AddGoogleTest) # Your existing module for GTest

# --- Conditional Libraries ---
# The FFT_mvec library is already configured for CUDA/HIP.
# We need to select the correct GPU-specific runtime libraries.
if(BUILD_WITH_HIP)
    set(TEST_GPU_LIBS
        roc::rocfft # Link if tests directly call rocfft APIs not covered by FFT_mvec
        roc::rocblas # Link if tests directly call rocblas APIs not covered by FFT_mvec
        roc::rccl # Link if tests directly call rccl APIs not covered by FFT_mvec
        hip::hipfft # Link if tests directly call hipfft APIs not covered by FFT_mvec
        roc::hipblas # Link if tests directly call hipblas APIs not covered by FFT_m
        ${HIP_LIBRARIES} # General HIP runtime from find_package(HIP)
    )

else() # CUDA Build
    if(USE_CUTENSOR)
        set(TEST_GPU_LIBS
            CUDA::cufft # Link if tests directly call cufft APIs not covered by FFT_mvec
            CUDA::cublas # Link if tests directly call cublas APIs not covered by FFT_mvec
            nccl # Link if tests directly call nccl APIs not covered by FFT_mvec
            cutensor # Link if tests directly call cutensor APIs not covered by FFT_mvec
            ${CUDAToolkit_LIBRARIES} # General CUDA runtime from find_package(CUDAToolkit)
        )
    else()
        set(TEST_GPU_LIBS
            CUDA::cufft # Link if tests directly call cufft APIs not covered by FFT_mvec
            CUDA::cublas # Link if tests directly call cublas APIs not covered by FFT_mvec
            nccl # Link if tests directly call nccl APIs not covered by FFT_mvec
            ${CUDAToolkit_LIBRARIES} # General CUDA runtime from find_package(CUDAToolkit)
        )
    endif()
endif()

# Common libraries for all tests
set(TEST_COMMON_LIBS
    FFT_mvec # Your main library, already dual-target
    OpenMP::OpenMP_CXX
    # OpenMP::OpenMP_C # If any C files with OpenMP are part of tests
    HighFive::Include # Assuming HighFive target is available
    hdf5::hdf5
    MPI::MPI_CXX # <--- ADD THIS
    MPI::MPI_C
    # MPI libraries are usually handled by MPIEXEC for running tests,
    # but if test code itself links MPI, add ${MPI_CXX_LIBRARIES}, ${MPI_C_LIBRARIES}
    # Example: ${MPI_CXX_LIBRARIES}
)

set(TEST_ALL_LIBS
    ${TEST_COMMON_LIBS}
    ${TEST_GPU_LIBS}
)

# pure C++ headers
set(TEST_PURE_CPP_HEADERS
    gtest-mpi-listener.hpp
)

# --- Test Source Files ---
# List all your primary test source files.
# We assume these might contain CUDA code and need hipification for the HIP path.
set(TEST_SOURCE_FILES_ORIGINAL
    test_Utils.cpp
    test_UtilKernels.cpp
    test_Comm.cpp
    test_Vector.cpp
    test_Matrix.cpp
    test_Matvec.cpp
)

set(TEST_FILES_TO_HIPIFY "") # Will store absolute paths
foreach(test_file ${TEST_SOURCE_FILES_ORIGINAL})
    list(APPEND TEST_FILES_TO_HIPIFY "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
endforeach()


# --- Hipification for Test Files (if BUILD_WITH_HIP is ON) ---
if(BUILD_WITH_HIP)
    set(ORIGINAL_CUDA_TEST_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    set(HIP_GENERATED_TEST_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipified)
    foreach(header_file ${TEST_PURE_CPP_HEADERS})
        configure_file("${ORIGINAL_CUDA_TEST_SRC_DIR}/${header_file}" "${HIP_GENERATED_TEST_SRC_DIR}/${header_file}")
    endforeach()
    file(COPY "${ORIGINAL_CUDA_TEST_SRC_DIR}/data" DESTINATION "${HIP_GENERATED_TEST_SRC_DIR}/")

    # TEST_FILES_TO_HIPIFY should be set here, containing full paths to original test source files
    # Example:
    # set(TEST_SOURCE_FILES_ORIGINAL test_Utils.cpp test_UtilKernels.cpp ...)
    # set(TEST_FILES_TO_HIPIFY "")
    # foreach(test_file ${TEST_SOURCE_FILES_ORIGINAL})
    #    list(APPEND TEST_FILES_TO_HIPIFY "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
    # endforeach()


    hipify_directory(
        INPUT_FILES_LIST "${TEST_FILES_TO_HIPIFY}" # Ensure this list is correctly populated
        BASE_CUDA_DIR "${ORIGINAL_CUDA_TEST_SRC_DIR}"
        OUTPUT_HIP_DIR "${HIP_GENERATED_TEST_SRC_DIR}"
        GENERATED_HIP_SOURCES_VAR "GENERATED_HIP_TEST_SOURCES"
        GENERATED_HIP_HEADERS_VAR "GENERATED_HIP_TEST_HEADERS"
    )
    # Ensure generated test headers (if any) can be found by generated test sources
    include_directories("${HIP_GENERATED_TEST_SRC_DIR} ${HIP_GENERATED_SRC_DIR_FOR_FFT_MVEC}")
else()
    include_directories(${CMAKE_SOURCE_DIR}/src)
endif()

# Include directories:
# - For finding headers from your 'src' directory (e.g., shared.hpp)
# - FFT_mvec should export its include directories (including hipified ones from src if applicable)
#   so linking against FFT_mvec makes its headers available.



# --- Define Tests ---
# Helper macro to get the correct source file (original or hipified)
macro(get_conditional_test_source VAR_NAME original_short_filename)
    if(BUILD_WITH_HIP)
        # Assuming hipify_directory keeps the original filename for .cpp files
        # within the HIP_GENERATED_TEST_SRC_DIR.
        # If your hipify_directory renames test_Foo.cpp to test_Foo.hip.cpp, adjust here.
        set(${VAR_NAME} "${HIP_GENERATED_TEST_SRC_DIR}/${original_short_filename}")
    else()
        set(${VAR_NAME} "${CMAKE_CURRENT_SOURCE_DIR}/${original_short_filename}")
    endif()
endmacro()

# Check for NUM_PROCS definition
if(NOT DEFINED NUM_PROCS)
    set(NUM_PROCS 2)
    message(STATUS "NUM_PROCS not defined for tests. Setting to 2.")
endif()

# Add each test using the conditional source file path
get_conditional_test_source(UTILS_TEST_SRC test_Utils.cpp)
add_gtest(UtilsTest "${UTILS_TEST_SRC}" "${TEST_ALL_LIBS}" 1)

get_conditional_test_source(UTILKERNELS_TEST_SRC test_UtilKernels.cpp)
add_gtest(UtilKernelsTest "${UTILKERNELS_TEST_SRC}" "${TEST_ALL_LIBS}" 1)

get_conditional_test_source(COMM_TEST_SRC test_Comm.cpp)
add_gtest(CommTest "${COMM_TEST_SRC}" "${TEST_ALL_LIBS}" ${NUM_PROCS})

get_conditional_test_source(VECTOR_TEST_SRC test_Vector.cpp)
add_gtest(VectorTest "${VECTOR_TEST_SRC}" "${TEST_ALL_LIBS}" ${NUM_PROCS})

get_conditional_test_source(MATRIX_TEST_SRC test_Matrix.cpp)
add_gtest(MatrixTest "${MATRIX_TEST_SRC}" "${TEST_ALL_LIBS}" ${NUM_PROCS})

get_conditional_test_source(MATVEC_TEST_SRC test_Matvec.cpp)
add_gtest(MatvecTest "${MATVEC_TEST_SRC}" "${TEST_ALL_LIBS}" ${NUM_PROCS})

# If you have a global HipifyAll target and want test hipification to be part of it:
if(BUILD_WITH_HIP AND TARGET HipifyAll)
    # This is tricky because HipifyAll is usually defined before add_subdirectory(test)
    # The dependency is better handled by the test executables depending on the OUTPUTS
    # of the add_custom_command calls made by hipify_directory for these test files.
    # CMake should handle this automatically if UTILS_TEST_SRC etc. are outputs of a custom command.
endif()
