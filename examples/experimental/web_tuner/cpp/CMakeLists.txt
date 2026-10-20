# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 4.0)
project(holoscan_web_tuner LANGUAGES CXX)

if(PROJECT_IS_TOP_LEVEL)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

# The experimental loopback tuner ships behind
# HOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER, which defaults to ON.
# Guard on the exported target so this file
# works both in-tree (where the alias
# holoscan::experimental_params_web_tuner exists — same gate idea as
# tests/params/CMakeLists.txt's `if(TARGET experimental_params_web_tuner)`)
# and against an installed SDK (where find_package imports the namespaced
# target only when the SDK was built with the flag ON).
if(TARGET holoscan::experimental_params_web_tuner)
    add_executable(web_tuner web_tuner.cpp)
    target_link_libraries(web_tuner
        PRIVATE
            holoscan::core
            holoscan::experimental_params_web_tuner
    )
    holoscan_configure_example_install(web_tuner)

    if(BUILD_TESTING AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # REQ: SYSTEM-015
        add_test(NAME EXAMPLE_CPP_WEB_TUNER_TEST COMMAND web_tuner)
        set_tests_properties(EXAMPLE_CPP_WEB_TUNER_TEST
            PROPERTIES
                PASS_REGULAR_EXPRESSION
                    "web_tuner: stopped accepted_edits=0 published_edits=0 catalog_schemas=2 samples=ok"
                LABELS "examples;parameter;runtime-update;experimental"
                TIMEOUT 30)
    endif()

    install(TARGETS web_tuner
        RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/experimental/web_tuner/cpp"
        COMPONENT holoscan-core-examples
    )
else()
    message(STATUS
        "web_tuner example excluded: the experimental web tuner was not built. "
        "Configure the SDK with -DHOLOSCAN_BUILD_EXPERIMENTAL_PARAMS_WEB_TUNER=ON to build it.")
endif()
