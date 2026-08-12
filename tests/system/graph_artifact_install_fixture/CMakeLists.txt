# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_graph_artifact_install_fixture LANGUAGES CXX)

foreach(required_variable IN ITEMS
        HOLOSCAN_GRAPH_ARTIFACTS_MODULE
        HOLOSCAN_ARTIFACT_FIXTURE_MODE)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()
if(NOT HOLOSCAN_ARTIFACT_FIXTURE_MODE MATCHES
   "^(extra_deployment|nested_directory|non_regular_file|valid_absolute_destination|default_install_excluded|cross_without_emulator)$")
    message(FATAL_ERROR
        "Unknown artifact fixture mode: ${HOLOSCAN_ARTIFACT_FIXTURE_MODE}")
endif()
if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "valid_absolute_destination" AND
   (NOT DEFINED HOLOSCAN_ABSOLUTE_INSTALL_DESTINATION OR
    HOLOSCAN_ABSOLUTE_INSTALL_DESTINATION STREQUAL ""))
    message(FATAL_ERROR "HOLOSCAN_ABSOLUTE_INSTALL_DESTINATION is required")
endif()

include("${HOLOSCAN_GRAPH_ARTIFACTS_MODULE}")

add_executable(artifact_install_fixture artifact_install_fixture.cpp)
target_compile_features(artifact_install_fixture PRIVATE cxx_std_20)
target_compile_definitions(artifact_install_fixture
    PRIVATE HOLOSCAN_ARTIFACT_FIXTURE_MODE="${HOLOSCAN_ARTIFACT_FIXTURE_MODE}"
)

set(artifact_install_destination share/graph-artifact-install-fixture)
if(HOLOSCAN_ARTIFACT_FIXTURE_MODE STREQUAL "valid_absolute_destination")
    set(artifact_install_destination "${HOLOSCAN_ABSOLUTE_INSTALL_DESTINATION}")
endif()

holoscan_add_graph_artifacts(
    TARGET artifact_install_fixture
    DEPLOYMENT_ID developer
    INSTALL_DESTINATION "${artifact_install_destination}"
)
