# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_EXAMPLE_HELPER
        HOLOSCAN_TEST_WORK_DIR
        HOLOSCAN_PYTHON_EXECUTABLE
        HOLOSCAN_GENERATOR
        HOLOSCAN_CTEST_COMMAND)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

file(REMOVE_RECURSE "${HOLOSCAN_TEST_WORK_DIR}")
set(test_source_dir "${HOLOSCAN_TEST_WORK_DIR}/source")
set(supported_build_dir "${HOLOSCAN_TEST_WORK_DIR}/supported")
file(MAKE_DIRECTORY "${test_source_dir}")
file(TO_CMAKE_PATH "${HOLOSCAN_EXAMPLE_HELPER}" example_helper)
file(TO_CMAKE_PATH "${HOLOSCAN_TEST_WORK_DIR}/package" example_package_dir)
file(WRITE "${test_source_dir}/probe.py" "print('python example interpreter probe')\n")
file(WRITE "${test_source_dir}/CMakeLists.txt"
    "cmake_minimum_required(VERSION 4.0)\n"
    "project(python_example_interpreter LANGUAGES NONE)\n"
    "include(CTest)\n"
    "set(HOLOSCAN_PYTHON_EXAMPLE_PACKAGE_DIR \"${example_package_dir}\")\n"
    "set(HOLOSCAN_EXAMPLES_IN_SDK_BUILD ON)\n"
    "include(\"${example_helper}\")\n"
    "holoscan_add_python_example(\n"
    "    NAME interpreter_probe\n"
    "    SCRIPT probe.py\n"
    "    TEST_NAME EXAMPLE_PYTHON_INTERPRETER_PROBE\n"
    ")\n"
)

set(supported_configure_command
    "${CMAKE_COMMAND}"
    -S "${test_source_dir}"
    -B "${supported_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DPython3_EXECUTABLE:FILEPATH=${HOLOSCAN_PYTHON_EXECUTABLE}"
)
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND supported_configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND supported_configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
execute_process(
    COMMAND ${supported_configure_command}
    RESULT_VARIABLE supported_result
    OUTPUT_VARIABLE supported_output
    ERROR_VARIABLE supported_error
    TIMEOUT 30
)
if(NOT supported_result EQUAL 0)
    message(FATAL_ERROR
        "A supported Python interpreter did not configure the example: "
        "${supported_output}${supported_error}")
endif()

execute_process(
    COMMAND "${HOLOSCAN_CTEST_COMMAND}" --test-dir "${supported_build_dir}" --show-only=json-v1
    RESULT_VARIABLE ctest_result
    OUTPUT_VARIABLE ctest_json
    ERROR_VARIABLE ctest_error
    TIMEOUT 15
)
if(NOT ctest_result EQUAL 0)
    message(FATAL_ERROR "Failed to inspect registered example tests: ${ctest_error}")
endif()

string(JSON test_count LENGTH "${ctest_json}" tests)
math(EXPR last_test_index "${test_count} - 1")
set(interpreter_registered FALSE)
foreach(test_index RANGE 0 ${last_test_index})
    string(JSON test_name GET "${ctest_json}" tests ${test_index} name)
    if(test_name STREQUAL "EXAMPLE_PYTHON_INTERPRETER_PROBE")
        string(JSON command_count LENGTH "${ctest_json}" tests ${test_index} command)
        math(EXPR last_command_index "${command_count} - 1")
        foreach(command_index RANGE 0 ${last_command_index})
            string(JSON command_argument GET
                "${ctest_json}" tests ${test_index} command ${command_index})
            if(command_argument STREQUAL HOLOSCAN_PYTHON_EXECUTABLE)
                set(interpreter_registered TRUE)
            endif()
        endforeach()
    endif()
endforeach()
if(NOT interpreter_registered)
    message(FATAL_ERROR
        "The Python example test did not register the selected supported interpreter: "
        "${ctest_json}")
endif()

execute_process(
    COMMAND "${HOLOSCAN_PYTHON_EXECUTABLE}" -c
        "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    RESULT_VARIABLE version_result
    OUTPUT_VARIABLE selected_version
    ERROR_VARIABLE version_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    TIMEOUT 15
)
if(NOT version_result EQUAL 0 OR selected_version VERSION_LESS 3.12)
    message(FATAL_ERROR
        "The supported example fixture selected Python ${selected_version}: ${version_error}")
endif()

if(UNIX)
    set(older_interpreter "${HOLOSCAN_TEST_WORK_DIR}/python3.11")
    file(WRITE "${older_interpreter}"
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-c\" ]; then\n"
        "    case \"$2\" in\n"
        "        *\"sys.version_info[:3]\"*) printf '%s' '3.11.9'; exit 0 ;;\n"
        "    esac\n"
        "fi\n"
        "exec \"${HOLOSCAN_PYTHON_EXECUTABLE}\" \"$@\"\n"
    )
    file(CHMOD "${older_interpreter}"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE
    )

    set(older_build_dir "${HOLOSCAN_TEST_WORK_DIR}/older")
    set(older_configure_command
        "${CMAKE_COMMAND}"
        -S "${test_source_dir}"
        -B "${older_build_dir}"
        -G "${HOLOSCAN_GENERATOR}"
        "-DPython3_EXECUTABLE:FILEPATH=${older_interpreter}"
    )
    if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
        list(APPEND older_configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
    endif()
    if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
        list(APPEND older_configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
    endif()
    execute_process(
        COMMAND ${older_configure_command}
        RESULT_VARIABLE older_result
        OUTPUT_VARIABLE older_output
        ERROR_VARIABLE older_error
        TIMEOUT 30
    )
    string(CONCAT older_log "${older_output}" "${older_error}")
    if(older_result EQUAL 0 OR NOT older_log MATCHES "3\\.12")
        message(FATAL_ERROR
            "An explicitly selected Python 3.11 interpreter was not rejected at the 3.12 floor: "
            "${older_log}")
    endif()
endif()
