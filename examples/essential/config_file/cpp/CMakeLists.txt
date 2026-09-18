# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_config_file LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

# In-tree compatibility: the SDK umbrella build permits the optional
# configuration/parameter targets to be disabled. Standalone installed-package
# builds use CMakeLists.min.txt and require the complete params_io closure.
if(NOT (TARGET holoscan::config_io AND TARGET holoscan::params AND TARGET holoscan::params_io))
    message(STATUS
        "config_file example excluded from the in-tree build: optional "
        "config/params targets are disabled.")
    return()
endif()

add_executable(config_file config_file.cpp)
target_link_libraries(config_file
    PRIVATE
        holoscan::core
        holoscan::config_io
        holoscan::params
        holoscan::params_io
)
# essential/<name>/cpp nests one level deeper than examples/<name>/cpp, so the
# installed binary needs one extra step up to reach lib/.
holoscan_configure_example_install(config_file)

if(BUILD_TESTING)
    # REQ: SYSTEM-015
    add_test(NAME EXAMPLE_CPP_CONFIG_FILE_TEST
        COMMAND config_file --config-dir ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(EXAMPLE_CPP_CONFIG_FILE_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "config_file: digests=match effective=\\(threshold=9 gain=2\\.5 window=8\\) duplicate=PARAM_CONFIG_FIELD_DUPLICATE roundtrip=\\(saved=3 contributed=3 retained=0\\) detections=3/6 termination=stopped"
            LABELS "examples;configuration;essential"
            TIMEOUT 30)
endif()

install(TARGETS config_file
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/essential/config_file/cpp"
    COMPONENT holoscan-core-examples
)
install(FILES detector.json detector.yaml
    DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/essential/config_file/cpp"
    COMPONENT holoscan-core-examples
)
