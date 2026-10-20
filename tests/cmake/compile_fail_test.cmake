# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

option(HOLOSCAN_ISOLATE_COMPILE_FAIL_TESTS
    "Run negative compilation probes outside the shared SDK build tree when supported" ON)
mark_as_advanced(HOLOSCAN_ISOLATE_COMPILE_FAIL_TESTS)

# Use the generator's exact commands rather than reconstructing transitive flags,
# includes, or language versions. Other toolchains keep the serialized build path.
set(_holoscan_isolate_compile_fail FALSE)
get_property(_holoscan_compile_fail_multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(HOLOSCAN_ISOLATE_COMPILE_FAIL_TESTS AND Python3_Interpreter_FOUND
        AND CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux"
        AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
        AND CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$"
        AND CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU"
        AND CMAKE_GENERATOR MATCHES "^(Ninja|Unix Makefiles)$"
        AND NOT _holoscan_compile_fail_multi_config)
    set(_holoscan_isolate_compile_fail TRUE)
endif()
unset(_holoscan_compile_fail_multi_config)

function(holoscan_compile_fail_dependencies)
    get_property(probes GLOBAL PROPERTY HOLOSCAN_COMPILE_FAIL_TARGETS)
    foreach(probe IN LISTS probes)
        # Read after all target customization (e.g. schema/sensor_io) is complete.
        get_target_property(dependencies "${probe}" LINK_LIBRARIES)
        foreach(dependency IN LISTS dependencies)
            if(NOT TARGET "${dependency}")
                message(FATAL_ERROR
                    "Compile-fail probe '${probe}' links '${dependency}', which is not a CMake target. "
                    "Isolated compile-fail probes must link only concrete CMake targets so their "
                    "prerequisites can be built before parallel compilation. Plain library names "
                    "and generator expressions are not supported here.")
            endif()
            add_dependencies(holoscan_compile_fail_prerequisites "${dependency}")
        endforeach()
    endforeach()
endfunction()

function(holoscan_add_compile_fail_test test_name target_name diagnostic)
    if(_holoscan_isolate_compile_fail)
        if(NOT TARGET holoscan_compile_fail_prerequisites)
            add_custom_target(holoscan_compile_fail_prerequisites)
            add_test(NAME COMPILE_FAIL.prepare
                COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
                    --config "$<CONFIG>" --target holoscan_compile_fail_prerequisites --parallel 2)
            set_tests_properties(COMPILE_FAIL.prepare PROPERTIES
                FIXTURES_SETUP compile_fail_dependencies RUN_SERIAL TRUE)
            cmake_language(DEFER DIRECTORY "${holoscan_SOURCE_DIR}/tests"
                CALL holoscan_compile_fail_dependencies)
        endif()
        set_property(GLOBAL APPEND PROPERTY HOLOSCAN_COMPILE_FAIL_TARGETS "${target_name}")
        set_target_properties("${target_name}" PROPERTIES
            EXPORT_COMPILE_COMMANDS ON UNITY_BUILD OFF)
        add_test(NAME "${test_name}"
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/verify_compile_failure.py"
                --database "${CMAKE_BINARY_DIR}/compile_commands.json"
                --target "${target_name}"
                --work-dir "${holoscan_BINARY_DIR}/compile-fail/${target_name}/$<CONFIG>"
                --diagnostic "${diagnostic}")
        set_tests_properties("${test_name}" PROPERTIES
            FIXTURES_REQUIRED compile_fail_dependencies PROCESSORS 1 TIMEOUT 300)
    else()
        add_test(NAME "${test_name}"
            COMMAND "${CMAKE_COMMAND}"
                "-DHOLOSCAN_BUILD_DIR=${holoscan_BINARY_DIR}"
                "-DCMAKE_BUILD_TYPE=$<CONFIG>"
                "-DHOLOSCAN_TARGET=${target_name}"
                "-DHOLOSCAN_EXPECTED_DIAGNOSTIC=${diagnostic}"
                -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../core/expected_compile_fail/verify_compile_failure.cmake")
        # MSVC, multi-config generators, and other unsupported frontends can rebuild
        # shared SDK libraries here, so they must still exclude every other CTest.
        set_tests_properties("${test_name}" PROPERTIES RUN_SERIAL TRUE)
    endif()
endfunction()

if(Python3_Interpreter_FOUND)
    add_test(NAME COMPILE_FAIL.runner_contract
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_LIST_DIR}/test_verify_compile_failure.py")
endif()
