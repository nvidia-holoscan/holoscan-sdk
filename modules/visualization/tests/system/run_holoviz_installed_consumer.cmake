# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Called after the operator consumer has prepared and relocated an isolated SDK installation.
# HoloViz has no package of its own: it ships beside HolovizOp in the module's
# holoscan-visualization-exports set (holoscan-visualization-targets.cmake, loaded by the SDK
# package's visualization component), and the operator consumer already scanned that package
# metadata for leaks.
file(GLOB_RECURSE stray_holoviz_package
    "${install_prefix}/*/cmake/holoviz/*.cmake"
    "${install_prefix}/*/*/cmake/holoviz/*.cmake")
file(GLOB_RECURSE stray_holoviz_archives
    "${install_prefix}/libholoviz.a" "${install_prefix}/holoviz.lib")
if(stray_holoviz_package OR stray_holoviz_archives)
    message(FATAL_ERROR
        "HoloViz must not install a standalone package or static archive: "
        "${stray_holoviz_package};${stray_holoviz_archives}")
endif()

foreach(discovery_mode IN ITEMS component unqualified)
    set(consumer_build_dir "${package_work_dir}/consumer_${discovery_mode}")
    # Reuse the operator consumer's generator, compiler, configuration and toolchain arguments.
    set(consumer_configure_command
        "${CMAKE_COMMAND}"
        ${consumer_configure_options}
        -S "${HOLOVIZ_CONSUMER_SOURCE_DIR}"
        -B "${consumer_build_dir}"
        "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
        "-DHOLOVIZ_DISCOVERY_MODE:STRING=${discovery_mode}"
    )
    execute_process(COMMAND ${consumer_configure_command} COMMAND_ERROR_IS_FATAL ANY)
    set(consumer_build_command "${CMAKE_COMMAND}" --build "${consumer_build_dir}" --verbose)
    set(consumer_test_command
        "${HOLOSCAN_CTEST_COMMAND}" --test-dir "${consumer_build_dir}" --output-on-failure)
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND consumer_build_command --config "${CMAKE_BUILD_TYPE}")
        list(APPEND consumer_test_command -C "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${consumer_build_command}
        OUTPUT_VARIABLE build_stdout ERROR_VARIABLE build_stderr RESULT_VARIABLE build_result)
    if(NOT build_result EQUAL 0)
        message(FATAL_ERROR "Failed to build HoloViz consumer:\n${build_stdout}\n${build_stderr}")
    endif()
    string(REPLACE "${HOLOVIZ_CONSUMER_SOURCE_DIR}" "<consumer-source>"
        build_output "${build_stdout}\n${build_stderr}")
    assert_no_sdk_source_path_leak("${build_output}" "HoloViz consumer build")
    execute_process(COMMAND ${consumer_test_command} COMMAND_ERROR_IS_FATAL ANY)
endforeach()
