# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_restart_control LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(restart_control restart_control.cpp)
target_link_libraries(restart_control PRIVATE holoscan::core)
holoscan_configure_example_install(restart_control)

if(BUILD_TESTING)
    # REQ: SYSTEM-015
    add_test(NAME EXAMPLE_CPP_RESTART_CONTROL_TEST COMMAND restart_control)
    set_tests_properties(EXAMPLE_CPP_RESTART_CONTROL_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION
                "restart_control: restarts published \\(1 self, 1 external\\); unauthorized target rejected \\(kLifecycleRestartUnauthorized\\); sink received 9 samples across 3 epochs termination=stopped"
            LABELS "examples;lifecycle;experimental"
            TIMEOUT 30)
endif()

install(TARGETS restart_control
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/experimental/restart_control/cpp"
    COMPONENT holoscan-core-examples
)
