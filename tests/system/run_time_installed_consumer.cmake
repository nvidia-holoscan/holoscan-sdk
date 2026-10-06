# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/consumer_coverage.cmake")

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_CONSUMER_SOURCE_DIR
        HOLOSCAN_CONSUMER_WORK_DIR
        HOLOSCAN_GENERATOR
        HOLOSCAN_CXX_COMPILER
        HOLOSCAN_CTEST_COMMAND)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(install_prefix "${HOLOSCAN_CONSUMER_WORK_DIR}/install")
set(consumer_build_dir "${HOLOSCAN_CONSUMER_WORK_DIR}/build")
set(installed_flatc_name "flatc")
if(WIN32)
    string(APPEND installed_flatc_name ".exe")
endif()
file(REMOVE_RECURSE "${install_prefix}" "${consumer_build_dir}")
file(MAKE_DIRECTORY "${consumer_build_dir}")
holoscan_exclude_consumer_coverage("${HOLOSCAN_BUILD_DIR}" "${consumer_build_dir}")

# sensor_io belongs here even though this consumer never names a sensor type. The install exports one
# holoscan-targets.cmake covering every exported target, and find_package fails outright if any of the
# files it references is missing -- so omitting a target here breaks the consumer for a reason that
# looks nothing like the actual cause.
set(sdk_build_targets core holoscan_schema sensor_io config config_io params params_io)
if(DEFINED HOLOSCAN_BUILD_SHARED_LIBS AND NOT HOLOSCAN_BUILD_SHARED_LIBS)
    # Static package installation exports the compiled spdlog archive even though Core consumes
    # only its header-only target. Build that installed dependency in this intentionally focused
    # target set before installing the holoscan-dependencies component.
    list(APPEND sdk_build_targets spdlog)
endif()
if(DEFINED HOLOSCAN_EXTRA_BUILD_TARGETS)
    list(APPEND sdk_build_targets ${HOLOSCAN_EXTRA_BUILD_TARGETS})
endif()
list(REMOVE_DUPLICATES sdk_build_targets)
set(sdk_build_command
    "${CMAKE_COMMAND}" --build "${HOLOSCAN_BUILD_DIR}" --target ${sdk_build_targets} --parallel 4
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND sdk_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${sdk_build_command} RESULT_VARIABLE sdk_build_result)
if(NOT sdk_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build SDK targets required by the installed-package consumer")
endif()

# holoscan-config and holoscan-config-io belong to the base set rather than arriving through
# HOLOSCAN_EXTRA_INSTALL_COMPONENTS: holoscan-core exports targets that link them, so every
# Core-bearing prefix needs them. The extra-components hook stays for a module that is optional
# and wanted by a single fixture.
set(install_components
    holoscan
    holoscan-dependencies
    iceoryx2-lib
    dev
    holoscan-holomq
    holoscan-core
    holoscan-sensor-io
    holoscan-config
    holoscan-config-io
)
if(DEFINED HOLOSCAN_EXTRA_INSTALL_COMPONENTS)
    list(APPEND install_components ${HOLOSCAN_EXTRA_INSTALL_COMPONENTS})
endif()
list(REMOVE_DUPLICATES install_components)
foreach(component IN LISTS install_components)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${install_prefix}"
        --component "${component}"
    )
    if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${CMAKE_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endforeach()
unset(install_components)

foreach(required_install_path IN ITEMS
        "${install_prefix}/bin/${installed_flatc_name}"
        "${install_prefix}/include/holoscan/core/managed_shutdown_signals.hpp"
        "${install_prefix}/include/holoscan/core/source_id.hpp"
        "${install_prefix}/include/holoscan/core/tag.hpp"
        "${install_prefix}/include/holoscan/time/clock_registry.hpp"
        "${install_prefix}/include/holoscan/time/time_point.hpp"
        "${install_prefix}/lib/cmake/holoscan/HoloscanSchemaTools.cmake"
        "${install_prefix}/lib/cmake/holoscan/generate_schema_traits.py"
        "${install_prefix}/lib/cmake/holoscan/holoscan-config.cmake"
        "${install_prefix}/share/holoscan/schemas/flatbuffers/reflection.fbs"
        "${install_prefix}/share/holoscan/schemas/tensor.fbs"
        "${install_prefix}/include/holoscan/sensor_io/schemas/image_generated.h"
        "${install_prefix}/share/holoscan/schemas/sensor_io/image.fbs")
    if(NOT EXISTS "${required_install_path}")
        message(FATAL_ERROR "Installed consumer prerequisite is missing: ${required_install_path}")
    endif()
endforeach()
if(EXISTS "${install_prefix}/include/holoscan/schemas/tensor_bfbs_generated.h")
    message(FATAL_ERROR "Default SDK schema generation unexpectedly installed runtime BFBS bytes")
endif()

set(configure_command
    "${CMAKE_COMMAND}"
    -S "${HOLOSCAN_CONSUMER_SOURCE_DIR}"
    -B "${consumer_build_dir}"
    -G "${HOLOSCAN_GENERATOR}"
    "-DHOLOSCAN_TEST_INSTALL_PREFIX:PATH=${install_prefix}"
    "-DCMAKE_PREFIX_PATH:PATH=${install_prefix}"
    "-Dholoscan_DIR:PATH=${install_prefix}/lib/cmake/holoscan"
    "-DCMAKE_CXX_COMPILER:FILEPATH=${HOLOSCAN_CXX_COMPILER}"
)
if(DEFINED HOLOSCAN_GENERATOR_PLATFORM AND NOT HOLOSCAN_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND configure_command -A "${HOLOSCAN_GENERATOR_PLATFORM}")
endif()
if(DEFINED HOLOSCAN_GENERATOR_TOOLSET AND NOT HOLOSCAN_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND configure_command -T "${HOLOSCAN_GENERATOR_TOOLSET}")
endif()
if(DEFINED HOLOSCAN_MAKE_PROGRAM AND NOT HOLOSCAN_MAKE_PROGRAM STREQUAL "")
    list(APPEND configure_command "-DCMAKE_MAKE_PROGRAM:FILEPATH=${HOLOSCAN_MAKE_PROGRAM}")
endif()
if(DEFINED HOLOSCAN_TOOLCHAIN_FILE AND NOT HOLOSCAN_TOOLCHAIN_FILE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_TOOLCHAIN_FILE:FILEPATH=${HOLOSCAN_TOOLCHAIN_FILE}")
endif()
if(DEFINED HOLOSCAN_MSVC_RUNTIME_LIBRARY AND NOT HOLOSCAN_MSVC_RUNTIME_LIBRARY STREQUAL "")
    list(APPEND configure_command
        "-DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${HOLOSCAN_MSVC_RUNTIME_LIBRARY}")
endif()
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND configure_command "-DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}")
endif()
if(DEFINED HOLOSCAN_CONSUMER_FIND_MODE AND NOT HOLOSCAN_CONSUMER_FIND_MODE STREQUAL "")
    list(APPEND configure_command
        "-DHOLOSCAN_PACKAGE_FIND_MODE:STRING=${HOLOSCAN_CONSUMER_FIND_MODE}")
endif()
if(DEFINED HOLOSCAN_MODULE_PACKAGE_DIR AND NOT HOLOSCAN_MODULE_PACKAGE_DIR STREQUAL "")
    list(APPEND configure_command
        "-DHOLOSCAN_MODULE_PACKAGE_DIR:PATH=${install_prefix}/${HOLOSCAN_MODULE_PACKAGE_DIR}")
endif()

execute_process(COMMAND ${configure_command} RESULT_VARIABLE configure_result)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR "Failed to configure the isolated installed-package consumer")
endif()

set(consumer_build_command
    "${CMAKE_COMMAND}" --build "${consumer_build_dir}" --parallel 4)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND consumer_build_command --config "${CMAKE_BUILD_TYPE}")
endif()
execute_process(COMMAND ${consumer_build_command} RESULT_VARIABLE consumer_build_result)
if(NOT consumer_build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build the isolated installed-package consumer")
endif()

set(consumer_test_command
    "${HOLOSCAN_CTEST_COMMAND}" --test-dir "${consumer_build_dir}" --output-on-failure --verbose
)
if(DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE STREQUAL "")
    list(APPEND consumer_test_command -C "${CMAKE_BUILD_TYPE}")
endif()
execute_process(
    COMMAND ${consumer_test_command}
    RESULT_VARIABLE consumer_test_result
    OUTPUT_VARIABLE consumer_test_stdout
    ERROR_VARIABLE consumer_test_stderr
)
if(NOT consumer_test_result EQUAL 0)
    message(FATAL_ERROR
        "Installed-package consumer execution failed:\n"
        "${consumer_test_stdout}\n${consumer_test_stderr}")
endif()

if(DEFINED HOLOSCAN_EXPECT_TERMINAL_SUMMARY_COUNT)
    # This private installed-consumer sentinel deliberately observes the current logger prefix
    # because no structured test sink is available across the package boundary. The prefix is not
    # a public API; update this fixture if the private terminal-summary rendering changes.
    set(consumer_test_output "${consumer_test_stdout}\n${consumer_test_stderr}")
    string(REGEX MATCHALL "RunSession failed:" terminal_summaries "${consumer_test_output}")
    list(LENGTH terminal_summaries terminal_summary_count)
    if(NOT terminal_summary_count EQUAL HOLOSCAN_EXPECT_TERMINAL_SUMMARY_COUNT)
        message(FATAL_ERROR
            "Installed runtime consumer emitted ${terminal_summary_count} terminal summaries; "
            "expected ${HOLOSCAN_EXPECT_TERMINAL_SUMMARY_COUNT}:\n${consumer_test_output}")
    endif()
endif()
