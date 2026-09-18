# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")

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
if(HOLOSCAN_BUILD_PARAMS)
    add_subdirectory(params)
endif()
add_subdirectory(holomq)
