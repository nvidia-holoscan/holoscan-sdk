# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Including this installed module has no Python, CUDA, or filesystem side effects.
include_guard(GLOBAL)

function(_holoscan_generate_stubgen_command output)
    set(options "")
    set(one_value_keywords
        PYTHON_EXECUTABLE
        MODE
        STUB_LIBRARY
        SCRATCH_DIRECTORY
    )
    set(multi_value_keywords
        PYTHON_PATH
        EXTENSION_TARGETS
        COMMAND
        ENVIRONMENT
    )
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        "${options}"
        "${one_value_keywords}"
        "${multi_value_keywords}"
    )
    set(command "${CMAKE_COMMAND}" -E env)
    if(ARG_PYTHON_PATH)
        if(CMAKE_HOST_WIN32)
            list(JOIN ARG_PYTHON_PATH ";" python_path)
            string(REPLACE ";" "\\;" python_path "${python_path}")
        else()
            list(JOIN ARG_PYTHON_PATH ":" python_path)
        endif()
        list(APPEND command "PYTHONPATH=${python_path}")
    endif()
    list(APPEND command "${ARG_PYTHON_EXECUTABLE}"
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/stubgen/run_stubgen.py"
        --mode "${ARG_MODE}" --scratch-dir "${ARG_SCRATCH_DIRECTORY}")
    if(ARG_STUB_LIBRARY)
        list(APPEND command --stub-library "${ARG_STUB_LIBRARY}")
    endif()
    foreach(target IN LISTS ARG_EXTENSION_TARGETS)
        list(APPEND command --extension "$<TARGET_FILE:${target}>")
    endforeach()
    list(APPEND command --)
    if(ARG_ENVIRONMENT)
        list(APPEND command "${CMAKE_COMMAND}" -E env ${ARG_ENVIRONMENT})
    endif()
    list(APPEND command ${ARG_COMMAND})
    set(${output} "${command}" PARENT_SCOPE)
endfunction()

# Declare one generation rule and a target that owns its outputs. No installation is implied.
# TARGET, OUTPUTS, PYTHON_EXECUTABLE, COMMAND, and EXTENSION_TARGETS are required.
# COMMAND is a complete argv list, including the generator's executable. Use the same Python
# as PYTHON_EXECUTABLE when the command runs Python. PYTHON_PATH replaces PYTHONPATH only for
# this rule; omit it to inherit the environment. Paths should be absolute; relative OUTPUTS,
# BYPRODUCTS, and WORKING_DIRECTORY are interpreted in the caller's binary directory.
# ENVIRONMENT supplies name=value entries to the generator only, not the loader probe.
#
# CUDA_MODE and CUDA_STUB_LIBRARY override the shared HOLOSCAN_PYTHON_STUBGEN_CUDA_* cache
# defaults per invocation. Without an explicit stub path, search only an already-selected
# CUDAToolkit. This module never finds/downloads Python, nanobind, or CUDA on the caller's behalf.
# ALL is opt-in. DEPENDS adds generator scripts and other inputs; extension targets, the runner,
# and the per-rule policy file are automatic file-level dependencies, including for Make.
function(holoscan_add_stub_generation)
    set(options ALL)
    set(one_value_keywords
        TARGET
        PYTHON_EXECUTABLE
        WORKING_DIRECTORY
        CUDA_MODE
        CUDA_STUB_LIBRARY
        COMMENT
    )
    set(multi_value_keywords
        OUTPUTS
        BYPRODUCTS
        EXTENSION_TARGETS
        PYTHON_PATH
        COMMAND
        DEPENDS
        ENVIRONMENT
    )
    cmake_parse_arguments(PARSE_ARGV 0 ARG
        "${options}"
        "${one_value_keywords}"
        "${multi_value_keywords}"
    )
    if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Invalid arguments to holoscan_add_stub_generation")
    endif()
    foreach(required TARGET PYTHON_EXECUTABLE OUTPUTS COMMAND EXTENSION_TARGETS)
        if(NOT ARG_${required})
            message(FATAL_ERROR "holoscan_add_stub_generation requires ${required}")
        endif()
    endforeach()
    if(NOT ARG_TARGET MATCHES "^[A-Za-z0-9_.+-]+$" OR TARGET "${ARG_TARGET}")
        message(FATAL_ERROR "Invalid or duplicate stub generation target: ${ARG_TARGET}")
    endif()
    if(NOT IS_ABSOLUTE "${ARG_PYTHON_EXECUTABLE}" OR NOT EXISTS "${ARG_PYTHON_EXECUTABLE}")
        message(FATAL_ERROR "PYTHON_EXECUTABLE must name the existing absolute generator interpreter")
    endif()
    foreach(target IN LISTS ARG_EXTENSION_TARGETS)
        if(NOT TARGET "${target}")
            message(FATAL_ERROR "Unknown stub generation extension target: ${target}")
        endif()
        get_target_property(target_type "${target}" TYPE)
        if(NOT target_type MATCHES "^(MODULE|SHARED)_LIBRARY$")
            message(FATAL_ERROR "Stub generation extension must be a shared/module library: ${target}")
        endif()
    endforeach()

    set(HOLOSCAN_PYTHON_STUBGEN_CUDA_MODE "auto" CACHE STRING
        "Typing generator CUDA loader policy: auto, native, or stub")
    set_property(CACHE HOLOSCAN_PYTHON_STUBGEN_CUDA_MODE PROPERTY STRINGS auto native stub)
    set(HOLOSCAN_PYTHON_STUBGEN_CUDA_STUB_LIBRARY "" CACHE FILEPATH
        "Optional CUDA toolkit driver stub used only by Python typing generators")
    if(NOT DEFINED ARG_CUDA_MODE)
        set(ARG_CUDA_MODE "${HOLOSCAN_PYTHON_STUBGEN_CUDA_MODE}")
    endif()
    if(NOT ARG_CUDA_MODE MATCHES "^(auto|native|stub)$")
        message(FATAL_ERROR "Invalid stub generation CUDA_MODE: ${ARG_CUDA_MODE}")
    endif()
    if(NOT DEFINED ARG_CUDA_STUB_LIBRARY)
        set(ARG_CUDA_STUB_LIBRARY "${HOLOSCAN_PYTHON_STUBGEN_CUDA_STUB_LIBRARY}")
    endif()
    if(NOT ARG_CUDA_STUB_LIBRARY AND NOT ARG_CUDA_MODE STREQUAL "native")
        set(stub_paths)
        foreach(root CUDAToolkit_TARGET_DIR CUDAToolkit_LIBRARY_ROOT)
            if(DEFINED ${root} AND NOT "${${root}}" STREQUAL "")
                list(APPEND stub_paths "${${root}}/lib/stubs" "${${root}}/lib64/stubs")
            endif()
        endforeach()
        if(CUDAToolkit_LIBRARY_DIR)
            list(APPEND stub_paths "${CUDAToolkit_LIBRARY_DIR}/stubs")
        endif()
        if(stub_paths)
            find_file(selected_stub libcuda.so PATHS ${stub_paths} NO_DEFAULT_PATH NO_CACHE)
            if(selected_stub)
                set(ARG_CUDA_STUB_LIBRARY "${selected_stub}")
            endif()
        endif()
    endif()

    foreach(paths OUTPUTS BYPRODUCTS)
        set(absolute_paths)
        foreach(path IN LISTS ARG_${paths})
            get_filename_component(path "${path}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
            list(APPEND absolute_paths "${path}")
        endforeach()
        set(ARG_${paths} "${absolute_paths}")
    endforeach()
    if(NOT ARG_WORKING_DIRECTORY)
        set(ARG_WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
    get_filename_component(ARG_WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(rule_dir "${CMAKE_CURRENT_BINARY_DIR}/holoscan-stubgen/${ARG_TARGET}")
    file(MAKE_DIRECTORY "${rule_dir}")
    file(CONFIGURE OUTPUT "${rule_dir}/policy.txt" CONTENT
        "mode=${ARG_CUDA_MODE}\nstub=${ARG_CUDA_STUB_LIBRARY}\npython=${ARG_PYTHON_EXECUTABLE}\npaths=${ARG_PYTHON_PATH}\ncommand=${ARG_COMMAND}\nenv=${ARG_ENVIRONMENT}\nworkdir=${ARG_WORKING_DIRECTORY}\n")
    set(stub_option)
    if(ARG_CUDA_STUB_LIBRARY)
        list(APPEND stub_option STUB_LIBRARY "${ARG_CUDA_STUB_LIBRARY}")
    endif()
    _holoscan_generate_stubgen_command(command
        PYTHON_EXECUTABLE "${ARG_PYTHON_EXECUTABLE}"
        MODE "${ARG_CUDA_MODE}" ${stub_option}
        SCRATCH_DIRECTORY "${rule_dir}/loader"
        PYTHON_PATH ${ARG_PYTHON_PATH} EXTENSION_TARGETS ${ARG_EXTENSION_TARGETS}
        ENVIRONMENT ${ARG_ENVIRONMENT} COMMAND ${ARG_COMMAND})
    set(output_directories "${ARG_WORKING_DIRECTORY}")
    foreach(path IN LISTS ARG_OUTPUTS ARG_BYPRODUCTS)
        get_filename_component(directory "${path}" DIRECTORY)
        list(APPEND output_directories "${directory}")
    endforeach()
    list(REMOVE_DUPLICATES output_directories)
    add_custom_command(
        OUTPUT ${ARG_OUTPUTS}
        BYPRODUCTS ${ARG_BYPRODUCTS}
        COMMAND "${CMAKE_COMMAND}" -E make_directory ${output_directories}
        COMMAND "${CMAKE_COMMAND}" -E chdir "${ARG_WORKING_DIRECTORY}" ${command}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        DEPENDS ${ARG_DEPENDS} ${ARG_EXTENSION_TARGETS} "${rule_dir}/policy.txt"
            "${CMAKE_CURRENT_FUNCTION_LIST_FILE}"
            "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/stubgen/run_stubgen.py"
        COMMENT "${ARG_COMMENT}"
        VERBATIM
    )
    set(all_option)
    if(ARG_ALL)
        set(all_option ALL)
    endif()
    add_custom_target("${ARG_TARGET}" ${all_option} DEPENDS ${ARG_OUTPUTS})
endfunction()

# Nanobind convenience layer. GENERATOR_SCRIPT optionally selects the caller's fetched/pinned
# stubgen.py. Otherwise locate nanobind.stubgen in PYTHON_EXECUTABLE's existing environment.
# No NB_DIR or SDK-private targets are assumed. MODULE and OUTPUT describe one facade;
# remaining arguments have the same meaning as in holoscan_add_stub_generation.
function(holoscan_add_nanobind_stubs)
    set(options ALL)
    set(one_value_keywords
        TARGET
        MODULE
        OUTPUT
        PYTHON_EXECUTABLE
        GENERATOR_SCRIPT
        WORKING_DIRECTORY
        CUDA_MODE
        CUDA_STUB_LIBRARY
        COMMENT
    )
    set(multi_value_keywords
        EXTENSION_TARGETS
        PYTHON_PATH
        DEPENDS
    )
    cmake_parse_arguments(PARSE_ARGV 0 ARG
        "${options}"
        "${one_value_keywords}"
        "${multi_value_keywords}"
    )
    if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES
        OR NOT ARG_MODULE OR NOT ARG_OUTPUT OR NOT ARG_PYTHON_EXECUTABLE)
        message(FATAL_ERROR "Invalid arguments to holoscan_add_nanobind_stubs")
    endif()
    if(NOT ARG_GENERATOR_SCRIPT)
        execute_process(
            COMMAND "${ARG_PYTHON_EXECUTABLE}" -c
                "import importlib.util; print(importlib.util.find_spec('nanobind.stubgen').origin)"
            RESULT_VARIABLE result OUTPUT_VARIABLE ARG_GENERATOR_SCRIPT
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Cannot locate nanobind in the selected Python: ${error}")
        endif()
    endif()
    if(NOT IS_ABSOLUTE "${ARG_GENERATOR_SCRIPT}" OR NOT EXISTS "${ARG_GENERATOR_SCRIPT}")
        message(FATAL_ERROR "GENERATOR_SCRIPT must name an existing absolute nanobind stubgen.py")
    endif()
    get_filename_component(ARG_OUTPUT "${ARG_OUTPUT}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    set(command "${ARG_PYTHON_EXECUTABLE}")
    if(COMMAND nanobind_sanitizer_preload_env)
        nanobind_sanitizer_preload_env(sanitizer_env ${ARG_EXTENSION_TARGETS})
        if(sanitizer_env)
            set(command "${CMAKE_COMMAND}" -E env "${sanitizer_env}")
            if(sanitizer_env MATCHES "asan")
                list(APPEND command "ASAN_OPTIONS=detect_leaks=0")
            endif()
            list(APPEND command "${ARG_PYTHON_EXECUTABLE}")
        endif()
    endif()
    set(forward)
    foreach(key TARGET PYTHON_EXECUTABLE WORKING_DIRECTORY CUDA_MODE CUDA_STUB_LIBRARY COMMENT
        EXTENSION_TARGETS PYTHON_PATH)
        if(DEFINED ARG_${key})
            list(APPEND forward "${key}" "${ARG_${key}}")
        endif()
    endforeach()
    if(ARG_ALL)
        list(APPEND forward ALL)
    endif()
    holoscan_add_stub_generation(${forward}
        OUTPUTS "${ARG_OUTPUT}" DEPENDS ${ARG_DEPENDS} "${ARG_GENERATOR_SCRIPT}"
        COMMAND ${command} "${ARG_GENERATOR_SCRIPT}" -q -m "${ARG_MODULE}" -o "${ARG_OUTPUT}")
endfunction()
