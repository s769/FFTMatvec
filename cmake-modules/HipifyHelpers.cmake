# cmake-modules/HipifyHelpers.cmake

#--------------------------------------------------------------------------------------------------
# function hipify_directory
#
# Converts CUDA files to HIP files using hipify-clang.
#
# ARGS:
#   BASE_CUDA_DIR              : Base directory of the original CUDA source files.
#                                Used to determine relative paths for output structure.
#   OUTPUT_HIP_DIR             : Directory where hipified files will be generated.
#   GENERATED_HIP_SOURCES_VAR  : Output variable name to store the list of generated HIP source files.
#   GENERATED_HIP_HEADERS_VAR  : Output variable name to store the list of generated HIP header files.
#   INPUT_FILES_LIST           : (multi-value) List of absolute paths to CUDA files to be hipified.
#   EXTRA_HIPIFY_INCLUDE_PATHS : (multi-value, optional) Extra include paths for hipify-clang.
#--------------------------------------------------------------------------------------------------
function(hipify_directory)
    # Define arguments
    set(options) # No boolean options
    set(oneValueArgs BASE_CUDA_DIR OUTPUT_HIP_DIR GENERATED_HIP_SOURCES_VAR GENERATED_HIP_HEADERS_VAR)
    set(multiValueArgs INPUT_FILES_LIST EXTRA_HIPIFY_INCLUDE_PATHS HIPIFY_CLANG_DEFINES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_BASE_CUDA_DIR OR NOT ARG_OUTPUT_HIP_DIR OR NOT ARG_GENERATED_HIP_SOURCES_VAR OR NOT ARG_GENERATED_HIP_HEADERS_VAR OR NOT ARG_INPUT_FILES_LIST)
        message(FATAL_ERROR "hipify_directory: Missing required arguments (BASE_CUDA_DIR, OUTPUT_HIP_DIR, GENERATED_HIP_SOURCES_VAR, GENERATED_HIP_HEADERS_VAR, INPUT_FILES_LIST).")
    endif()

    find_program(HIPIFY_CLANG_EXECUTABLE hipify-clang
        HINTS ENV ROCM_PATH /opt/rocm-6.4.0/bin /opt/rocm-6.4.0/hip/bin # Adjust if ROCM_PATH is not standard
        DOC "Path to hipify-clang executable")

    if(NOT HIPIFY_CLANG_EXECUTABLE)
        message(FATAL_ERROR "hipify-clang not found. Please ensure it's in your PATH or ROCM_PATH environment variable is set correctly and points to a ROCm installation.")
    endif()

    set(CURRENT_GENERATED_SOURCES "")
    set(CURRENT_GENERATED_HEADERS "")

    # Prepare include path arguments for hipify-clang
    set(HIPIFY_INCLUDE_ARGS "")
    # Add the base directory of the sources being hipified. This helps hipify-clang resolve relative includes.
    list(APPEND HIPIFY_INCLUDE_ARGS -I=${ARG_BASE_CUDA_DIR})
    # Add any explicitly provided extra include paths
    foreach(INCLUDE_PATH ${ARG_EXTRA_HIPIFY_INCLUDE_PATHS})
        list(APPEND HIPIFY_INCLUDE_ARGS -I=${INCLUDE_PATH})
        message(STATUS ${INCLUDE_PATH})
    endforeach()

    set(LOCAL_HIPIFY_CLANG_OPTIONS "") # Use a local variable for options like -D
    foreach(DEFINITION ${ARG_HIPIFY_CLANG_DEFINES})
        # Each item in HIPIFY_CLANG_DEFINES should be like "MACRO=VALUE" or just "MACRO"
        list(APPEND LOCAL_HIPIFY_CLANG_OPTIONS -D${DEFINITION})
    endforeach()

    # Add project's own include directories that might be needed for parsing
    # This requires careful management. For now, ARG_BASE_CUDA_DIR and EXTRA_HIPIFY_INCLUDE_PATHS are used.
    # If you have a global list of project includes, you might add them here too.
    # Example: get_property(PROJECT_INCLUDES DIRECTORY PROPERTY INCLUDE_DIRECTORIES)
    # foreach(PROJECT_INCLUDE_DIR ${PROJECT_INCLUDES})
    #    list(APPEND HIPIFY_INCLUDE_ARGS --include-path=${PROJECT_INCLUDE_DIR})
    # endforeach()


    foreach(INPUT_FILE_FULL_PATH ${ARG_INPUT_FILES_LIST})
        file(RELATIVE_PATH RELATIVE_INPUT_FILE "${ARG_BASE_CUDA_DIR}" "${INPUT_FILE_FULL_PATH}")

        get_filename_component(INPUT_FILE_NAME_WE ${RELATIVE_INPUT_FILE} NAME_WE) # Name without extension
        get_filename_component(INPUT_FILE_EXT ${RELATIVE_INPUT_FILE} EXT) # Extension (e.g., .cu, .cpp, .hpp)
        get_filename_component(INPUT_FILE_DIR_RELATIVE ${RELATIVE_INPUT_FILE} DIRECTORY) # Relative directory

        set(HIP_OUTPUT_SUBDIR "${ARG_OUTPUT_HIP_DIR}/${INPUT_FILE_DIR_RELATIVE}")
        file(MAKE_DIRECTORY "${HIP_OUTPUT_SUBDIR}") # Ensure output subdirectory exists

        # Determine output file name for hipified files
        set(OUTPUT_FILE_NAME_BASE "${INPUT_FILE_NAME_WE}")
        set(OUTPUT_FILE_EXTENSION "${INPUT_FILE_EXT}") # Default to original extension for headers/cpp

        if("${INPUT_FILE_EXT}" STREQUAL ".cu")
            set(OUTPUT_FILE_EXTENSION ".hip.cpp") # Convert .cu to .hip.cpp
        elseif("${INPUT_FILE_EXT}" STREQUAL ".cpp" OR "${INPUT_FILE_EXT}" STREQUAL ".cxx" OR "${INPUT_FILE_EXT}" STREQUAL ".cc")
            # For .cpp files containing CUDA, hipify-clang typically outputs .cpp.
            # To avoid clashes if source/output dirs are not strictly separate for some reason,
            # or for clarity, you could rename them to .hip.cpp as well.
            # However, if they are just C++ files with some CUDA types (often for headers),
            # keeping the .cpp extension might be fine if they are placed in a distinct HIP_OUTPUT_SUBDIR.
            # For simplicity, we'll keep .cpp for hipified .cpp files. Headers keep their names.
            # If a .cpp file is mostly device code, consider renaming it to .cu first.
            set(OUTPUT_FILE_EXTENSION "${INPUT_FILE_EXT}")
        endif()
        set(HIP_OUTPUT_FILE "${HIP_OUTPUT_SUBDIR}/${OUTPUT_FILE_NAME_BASE}${OUTPUT_FILE_EXTENSION}")



        add_custom_command(
            OUTPUT "${HIP_OUTPUT_FILE}"
            COMMAND ${HIPIFY_CLANG_EXECUTABLE}
            ${LOCAL_HIPIFY_CLANG_OPTIONS}
            ${HIPIFY_INCLUDE_ARGS} # Include paths for hipify-clang to parse files
            # --cuda-path=${CUDAToolkit_TOOLKIT_ROOT_DIR} # Might be needed if hipify-clang can't find CUDA headers
            # --rocminfo-path=path_to_rocminfo # If specific version needed
            # --rocm-path=${ROCM_PATH} # If hipify-clang needs it explicitly
            # Add other hipify-clang options as needed:
            # --no-output-style-macro (prevents #define __HIP_ROCclr__ 1 etc.)
            # --default-macro-visibility=hidden (for internal macros)
            # --add-hip-pch # If using precompiled headers for HIP
            "${INPUT_FILE_FULL_PATH}"
            -o "${HIP_OUTPUT_FILE}"
            --default-preprocessor
            -- -std=c++17
            DEPENDS "${INPUT_FILE_FULL_PATH}" # Re-run if the source CUDA file changes
            COMMENT "Hipifying ${INPUT_FILE_FULL_PATH} to ${HIP_OUTPUT_FILE}"
            VERBATIM # Crucial for commands with special characters or list arguments
        )

        # Classify generated file as source or header based on its new extension
        if("${HIP_OUTPUT_FILE}" MATCHES "\\.(cpp|cc|cxx|hip\\.cpp)$")
            list(APPEND CURRENT_GENERATED_SOURCES "${HIP_OUTPUT_FILE}")
        elseif("${HIP_OUTPUT_FILE}" MATCHES "\\.(h|hpp|hxx|cuh)$")
            list(APPEND CURRENT_GENERATED_HEADERS "${HIP_OUTPUT_FILE}")
        else()
            message(WARNING "Hipified file ${HIP_OUTPUT_FILE} with extension ${OUTPUT_FILE_EXTENSION} not classified as source or header.")
        endif()
    endforeach()

    set(${ARG_GENERATED_HIP_SOURCES_VAR} ${CURRENT_GENERATED_SOURCES} PARENT_SCOPE)
    set(${ARG_GENERATED_HIP_HEADERS_VAR} ${CURRENT_GENERATED_HEADERS} PARENT_SCOPE)
endfunction()
