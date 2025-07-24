# cmake-modules/HipifyHelpers.cmake

#--------------------------------------------------------------------------------------------------
# function hipify_directory
#
# Converts CUDA files to HIP files using hipify-perl.
#
# ARGS:
#   BASE_CUDA_DIR              : Base directory of the original CUDA source files.
#                                Used to determine relative paths for output structure.
#   OUTPUT_HIP_DIR             : Directory where hipified files will be generated.
#   GENERATED_HIP_SOURCES_VAR  : Output variable name to store the list of generated HIP source files.
#   GENERATED_HIP_HEADERS_VAR  : Output variable name to store the list of generated HIP header files.
#   INPUT_FILES_LIST           : (multi-value) List of absolute paths to CUDA files to be hipified.
#--------------------------------------------------------------------------------------------------
function(hipify_directory)
    # Define arguments
    set(options) # No boolean options
    set(oneValueArgs BASE_CUDA_DIR OUTPUT_HIP_DIR GENERATED_HIP_SOURCES_VAR GENERATED_HIP_HEADERS_VAR)
    set(multiValueArgs INPUT_FILES_LIST EXTRA_HIPIFY_INCLUDE_PATHS HIPIFY_PERL_DEFINES)
    cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT ARG_BASE_CUDA_DIR OR NOT ARG_OUTPUT_HIP_DIR OR NOT ARG_GENERATED_HIP_SOURCES_VAR OR NOT ARG_GENERATED_HIP_HEADERS_VAR OR NOT ARG_INPUT_FILES_LIST)
        message(FATAL_ERROR "hipify_directory: Missing required arguments (BASE_CUDA_DIR, OUTPUT_HIP_DIR, GENERATED_HIP_SOURCES_VAR, GENERATED_HIP_HEADERS_VAR, INPUT_FILES_LIST).")
    endif()

    find_program(HIPIFY_PERL_EXECUTABLE hipify-perl
        HINTS ENV ROCM_PATH /opt/rocm-6.4.0/bin /opt/rocm-6.4.0/hip/bin # Adjust if ROCM_PATH is not standard
        DOC "Path to hipify-perl executable")

    if(NOT HIPIFY_PERL_EXECUTABLE)
        message(FATAL_ERROR "hipify-perl not found. Please ensure it's in your PATH or ROCM_PATH environment variable is set correctly and points to a ROCm installation.")
    endif()

    set(CURRENT_GENERATED_SOURCES "")
    set(CURRENT_GENERATED_HEADERS "")


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
            # For .cpp files containing CUDA, hipify-perl typically outputs .cpp.
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
            COMMAND ${HIPIFY_PERL_EXECUTABLE}
            "${INPUT_FILE_FULL_PATH}"
            -o "${HIP_OUTPUT_FILE}"
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
