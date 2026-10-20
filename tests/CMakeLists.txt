# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")

# Apply the test optimization level (see holoscan_define_test_optimization) to every target defined
# below this directory. Libraries, benchmarks, and examples keep the build type's flags; module
# tests apply it per target with holoscan_apply_test_optimization().
holoscan_define_test_optimization()
if(HOLOSCAN_TEST_OPTIMIZATION)
    add_compile_options(
        "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CONFIG:Debug>>>:${HOLOSCAN_TEST_OPTIMIZATION}>")
endif()

function(holoscan_configure_test_clang_tidy target_name)
    if(HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY)
        # Keep high-signal clang-tidy checks on test targets, but disable checks made noisy by
        # GTest assertion macro expansions.
        set(_holoscan_test_clang_tidy_checks
            "${CLANG_TIDY_CHECKS},-readability-function-cognitive-complexity"
        )
        set_target_properties(${target_name}
            PROPERTIES
                CXX_CLANG_TIDY "${CMAKE_CXX_CLANG_TIDY};--checks=${_holoscan_test_clang_tidy_checks}"
        )
        unset(_holoscan_test_clang_tidy_checks)
    endif()
endfunction()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/compile_fail_test.cmake")

if(Python3_Interpreter_FOUND)
    add_test(NAME INSTALLED_CONSUMERS.preparation_contract
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_LIST_DIR}/cmake/test_consumer_preparation.py")
    add_test(NAME INSTALLED_CONSUMERS.external_expected
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_LIST_DIR}/cmake/test_external_expected.py")
    add_test(NAME INSTALLED_CONSUMERS.dependency_layout
        COMMAND "${Python3_EXECUTABLE}" -m pytest -q
            "${CMAKE_CURRENT_LIST_DIR}/cmake/test_installed_dependency_layout.py")
endif()

option(HOLOSCAN_TEST_PRECOMPILED_HEADERS
    "Precompile headers shared by most sources of the largest C++ test executables" ON)

# Precompile the given headers for the C++ sources of a test executable, so that each source skips
# parsing them. CUDA sources are unaffected. clang-tidy cannot read GCC precompiled headers, so
# clang-tidy builds compile without them, which also checks that each source includes what it uses.
function(holoscan_test_precompile_headers target_name)
    if(NOT HOLOSCAN_TEST_PRECOMPILED_HEADERS OR (HOLOSCAN_ENABLE_CLANG_TIDY AND CMAKE_CXX_CLANG_TIDY))
        return()
    endif()
    set(_holoscan_pch_headers)
    foreach(_holoscan_pch_header IN LISTS ARGN)
        list(APPEND _holoscan_pch_headers "$<$<COMPILE_LANGUAGE:CXX>:${_holoscan_pch_header}>")
    endforeach()
    target_precompile_headers(${target_name} PRIVATE ${_holoscan_pch_headers})
endfunction()

add_subdirectory(core)
add_subdirectory(authoring)
add_subdirectory(config)
add_subdirectory(channel)
add_subdirectory(compiler)
add_subdirectory(cuda_execution)
add_subdirectory(memory)
add_subdirectory(runtime)
add_subdirectory(sensor_io)
add_subdirectory(time)
add_subdirectory(system)
add_subdirectory(threading)
add_subdirectory(params)
add_subdirectory(holomq)
