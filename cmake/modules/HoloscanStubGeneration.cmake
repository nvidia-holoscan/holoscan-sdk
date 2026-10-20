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
# Each extension target may also declare HOLOSCAN_PYTHON_IMPORT_DEPENDENCIES (targets or files)
# needed to import its package. These prerequisites are added even for consumers in sibling
# directories, where the SDK's temporary dependency variables are no longer in scope.
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
        get_target_property(import_dependencies "${target}" HOLOSCAN_PYTHON_IMPORT_DEPENDENCIES)
        if(import_dependencies)
            list(APPEND ARG_DEPENDS ${import_dependencies})
        endif()
    endforeach()
    list(REMOVE_DUPLICATES ARG_DEPENDS)

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
# The SDK Python build sets HOLOSCAN_SDK_STUBGEN_DEPENDS for shared import prerequisites.
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
    if(HOLOSCAN_SDK_STUBGEN_DEPENDS)
        list(APPEND ARG_DEPENDS ${HOLOSCAN_SDK_STUBGEN_DEPENDS})
    endif()
    holoscan_add_stub_generation(${forward}
        OUTPUTS "${ARG_OUTPUT}" DEPENDS ${ARG_DEPENDS} "${ARG_GENERATOR_SCRIPT}"
        COMMAND ${command} "${ARG_GENERATOR_SCRIPT}" -q -m "${ARG_MODULE}" -o "${ARG_OUTPUT}")
endfunction()

# Set <output> to the import roots for the holoscan typing tools: the given roots, then the
# installed SDK's holoscan_PYTHON_PACKAGE_DIR. Both locating the tools and running the generator
# use this list, so the generator imports the holoscan package that it ships in.
function(_holoscan_python_tools_import_roots output)
    set(roots ${ARGN})
    if(holoscan_PYTHON_PACKAGE_DIR)
        list(APPEND roots "${holoscan_PYTHON_PACKAGE_DIR}")
    endif()
    list(REMOVE_DUPLICATES roots)
    set(${output} "${roots}" PARENT_SCOPE)
endfunction()

# Find the typing tools that ship in the holoscan Python package (holoscan/_tools), so a provider
# uses the generator matching the registry it imports. Sets <output> to that directory. The
# tools type Operators that native provider extensions register; Operators written as Python
# classes are typed by their own source and need neither these tools nor a native build.
#
# HOLOSCAN_PYTHON_TOOLS_DIR wins when set; SDK builds set it to the source copy. Otherwise this
# asks PYTHON_EXECUTABLE where `holoscan` is, searching PYTHON_PATH and then the installed SDK's
# holoscan_PYTHON_PACKAGE_DIR. With neither, the inherited PYTHONPATH and the interpreter's own
# environment are searched. Locating the package does not import it.
function(holoscan_python_tools_directory output)
    set(options "")
    set(one_value_keywords PYTHON_EXECUTABLE)
    set(multi_value_keywords PYTHON_PATH)
    cmake_parse_arguments(PARSE_ARGV 1 ARG
        "${options}"
        "${one_value_keywords}"
        "${multi_value_keywords}"
    )
    if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES OR NOT ARG_PYTHON_EXECUTABLE)
        message(FATAL_ERROR "Invalid arguments to holoscan_python_tools_directory")
    endif()
    if(HOLOSCAN_PYTHON_TOOLS_DIR)
        set(directory "${HOLOSCAN_PYTHON_TOOLS_DIR}")
    else()
        _holoscan_python_tools_import_roots(roots ${ARG_PYTHON_PATH})
        set(environment)
        if(roots)
            if(CMAKE_HOST_WIN32)
                list(JOIN roots ";" python_path)
                string(REPLACE ";" "\\;" python_path "${python_path}")
            else()
                list(JOIN roots ":" python_path)
            endif()
            set(environment "${CMAKE_COMMAND}" -E env "PYTHONPATH=${python_path}")
        endif()
        execute_process(
            COMMAND ${environment} "${ARG_PYTHON_EXECUTABLE}" -c
                "import importlib.util, pathlib; spec = importlib.util.find_spec('holoscan'); print(pathlib.Path(next(iter(spec.submodule_search_locations)), '_tools'))"
            RESULT_VARIABLE result OUTPUT_VARIABLE directory
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_VARIABLE error)
        if(NOT result EQUAL 0)
            message(FATAL_ERROR
                "Cannot locate the holoscan Python package for its typing tools. Pass the "
                "directory containing holoscan in PYTHON_PATH or set HOLOSCAN_PYTHON_TOOLS_DIR: "
                "${error}")
        endif()
    endif()
    if(NOT EXISTS "${directory}/generate_operator_stubs.py")
        message(FATAL_ERROR
            "${directory} does not contain the holoscan typing tools; this Holoscan SDK predates "
            "them or was built without Python")
    endif()
    set(${output} "${directory}" PARENT_SCOPE)
endfunction()

# Generate the typed proxy stub for a provider's registered Operators with the SDK's generator.
#
# Required:
#   TARGET: target that owns OUTPUT.
#   OUTPUT: generated .pyi path; relative paths are in the caller's binary directory.
#   PROVIDER_MODULES: Python modules imported to register the provider's rows.
#   REGISTRY_PREFIX: stable-name prefix selecting the provider's rows, for example holoscan.vision.
#   PYTHON_EXECUTABLE and EXTENSION_TARGETS: as for holoscan_add_stub_generation.
# Optional:
#   PYTHON_PATH: import roots of the provider and Core. The generator runs with these roots
#     followed by holoscan_PYTHON_PACKAGE_DIR, the same path that locates the tools; with
#     neither, it inherits PYTHONPATH.
#   HANDLE_ANNOTATIONS: TYPE_NAME=MODULE.CLASS entries typing constructor-handle keywords, where
#     MODULE.CLASS is a dotted Python path. Every handle type except the SDK's built-in
#     holoscan.VoidCallback needs one.
#   MODULE_DOC: docstring of the generated module.
#   HEADER_FILE: comment-only license header replacing the SDK's.
#   COMMITTED_STUB and CHECK_TARGET: add CHECK_TARGET, which fails unless the committed stub
#     declares every generated name, including the private parameter and proxy types, exactly
#     as generated. It may add declarations and export more names, so one committed stub can
#     cover build configurations that register different subsets of Operators. It is never part
#     of ALL.
#   ALL, DEPENDS, CUDA_MODE, CUDA_STUB_LIBRARY, COMMENT: as for holoscan_add_stub_generation.
function(holoscan_add_operator_stubs)
    set(options ALL)
    set(one_value_keywords
        TARGET
        OUTPUT
        REGISTRY_PREFIX
        PYTHON_EXECUTABLE
        MODULE_DOC
        HEADER_FILE
        COMMITTED_STUB
        CHECK_TARGET
        CUDA_MODE
        CUDA_STUB_LIBRARY
        COMMENT
    )
    set(multi_value_keywords
        PROVIDER_MODULES
        EXTENSION_TARGETS
        PYTHON_PATH
        HANDLE_ANNOTATIONS
        DEPENDS
    )
    cmake_parse_arguments(PARSE_ARGV 0 ARG
        "${options}"
        "${one_value_keywords}"
        "${multi_value_keywords}"
    )
    if(ARG_UNPARSED_ARGUMENTS OR ARG_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Invalid arguments to holoscan_add_operator_stubs")
    endif()
    foreach(required TARGET OUTPUT PROVIDER_MODULES REGISTRY_PREFIX PYTHON_EXECUTABLE
            EXTENSION_TARGETS)
        if(NOT ARG_${required})
            message(FATAL_ERROR "holoscan_add_operator_stubs requires ${required}")
        endif()
    endforeach()
    set(has_committed_stub FALSE)
    set(has_check_target FALSE)
    if(DEFINED ARG_COMMITTED_STUB)
        set(has_committed_stub TRUE)
    endif()
    if(DEFINED ARG_CHECK_TARGET)
        set(has_check_target TRUE)
    endif()
    if(NOT has_committed_stub STREQUAL has_check_target)
        message(FATAL_ERROR "holoscan_add_operator_stubs needs COMMITTED_STUB and CHECK_TARGET together")
    endif()
    foreach(mapping IN LISTS ARG_HANDLE_ANNOTATIONS)
        if(NOT mapping MATCHES "^[^=]+=[A-Za-z_][A-Za-z0-9_]*(\\.[A-Za-z_][A-Za-z0-9_]*)+$")
            message(FATAL_ERROR "HANDLE_ANNOTATIONS entries must be TYPE_NAME=MODULE.CLASS: ${mapping}")
        endif()
    endforeach()
    # Locate the tools and run the generator with one import path, so the generator imports the
    # holoscan package that it ships in.
    _holoscan_python_tools_import_roots(python_path ${ARG_PYTHON_PATH})
    set(python_path_option)
    if(python_path)
        set(python_path_option PYTHON_PATH ${python_path})
    endif()
    holoscan_python_tools_directory(tools
        PYTHON_EXECUTABLE "${ARG_PYTHON_EXECUTABLE}" ${python_path_option})
    get_filename_component(ARG_OUTPUT "${ARG_OUTPUT}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")

    set(command "${ARG_PYTHON_EXECUTABLE}" "${tools}/generate_operator_stubs.py"
        --output "${ARG_OUTPUT}" --registry-prefix "${ARG_REGISTRY_PREFIX}")
    foreach(module IN LISTS ARG_PROVIDER_MODULES)
        list(APPEND command --provider-module "${module}")
    endforeach()
    foreach(mapping IN LISTS ARG_HANDLE_ANNOTATIONS)
        list(APPEND command --handle-annotation "${mapping}")
    endforeach()
    if(DEFINED ARG_MODULE_DOC)
        list(APPEND command --module-doc "${ARG_MODULE_DOC}")
    endif()
    # The generator projects holoscan.registry, which sits beside the tools in every layout.
    set(depends ${ARG_DEPENDS} "${tools}/generate_operator_stubs.py")
    get_filename_component(registry "${tools}/../registry.py" ABSOLUTE)
    if(EXISTS "${registry}")
        list(APPEND depends "${registry}")
    endif()
    if(DEFINED ARG_HEADER_FILE)
        get_filename_component(ARG_HEADER_FILE "${ARG_HEADER_FILE}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        list(APPEND command --header-file "${ARG_HEADER_FILE}")
        list(APPEND depends "${ARG_HEADER_FILE}")
    endif()
    if(HOLOSCAN_SDK_STUBGEN_DEPENDS)
        list(APPEND depends ${HOLOSCAN_SDK_STUBGEN_DEPENDS})
    endif()
    set(forward)
    foreach(key TARGET PYTHON_EXECUTABLE CUDA_MODE CUDA_STUB_LIBRARY COMMENT)
        if(DEFINED ARG_${key})
            list(APPEND forward "${key}" "${ARG_${key}}")
        endif()
    endforeach()
    if(ARG_ALL)
        list(APPEND forward ALL)
    endif()
    holoscan_add_stub_generation(${forward} ${python_path_option}
        OUTPUTS "${ARG_OUTPUT}" EXTENSION_TARGETS ${ARG_EXTENSION_TARGETS}
        DEPENDS ${depends} COMMAND ${command})

    if(DEFINED ARG_CHECK_TARGET)
        if(TARGET "${ARG_CHECK_TARGET}")
            message(FATAL_ERROR "Duplicate operator stub check target: ${ARG_CHECK_TARGET}")
        endif()
        get_filename_component(ARG_COMMITTED_STUB "${ARG_COMMITTED_STUB}" ABSOLUTE
            BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
        add_custom_target("${ARG_CHECK_TARGET}"
            COMMAND "${ARG_PYTHON_EXECUTABLE}" "${tools}/check_stub_coverage.py"
                    --generated "${ARG_OUTPUT}" --committed "${ARG_COMMITTED_STUB}"
                    --match-declarations
            DEPENDS "${tools}/check_stub_coverage.py" "${ARG_COMMITTED_STUB}"
            COMMENT "Checking that ${ARG_COMMITTED_STUB} covers the generated Operator stub"
            VERBATIM
        )
        # Order after generation without sharing the output rule, which Make would run twice.
        add_dependencies("${ARG_CHECK_TARGET}" "${ARG_TARGET}")
    endif()
endfunction()
