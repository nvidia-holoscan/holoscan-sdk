# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 4.0)
    project(holoscan_camera_imu_control LANGUAGES CXX)
    include("${CMAKE_CURRENT_LIST_DIR}/CMakeLists.min.txt")
    return()
endif()

add_executable(camera_imu_control camera_imu_control.cpp)
target_compile_features(camera_imu_control PRIVATE cxx_std_20)
target_link_libraries(camera_imu_control PRIVATE holoscan::core Threads::Threads)
holoscan_configure_example_install(camera_imu_control)

holoscan_add_graph_artifacts(
    TARGET camera_imu_control
    DEPLOYMENT_ID developer
    OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/holoscan-artifacts"
    INSTALL_DESTINATION "${CMAKE_INSTALL_DATADIR}/holoscan/camera_imu_control/holoscan"
)

if(BUILD_TESTING)
    # REQ: SYSTEM-004
    add_test(NAME EXAMPLE_CAMERA_IMU_RUNTIME_TEST COMMAND camera_imu_control)
    set_tests_properties(EXAMPLE_CAMERA_IMU_RUNTIME_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "camera/IMU runtime complete"
            TIMEOUT 15
    )
    # REQ: SYSTEM-004
    add_test(NAME EXAMPLE_CAMERA_IMU_ARTIFACT_TEST
        COMMAND camera_imu_control
            --emit-artifacts
            --artifact-root "${CMAKE_CURRENT_BINARY_DIR}/test-artifacts"
            --deployment-id ctest
    )
    set_tests_properties(EXAMPLE_CAMERA_IMU_ARTIFACT_TEST
        PROPERTIES
            PASS_REGULAR_EXPRESSION "artifact-only publication complete"
            TIMEOUT 15
    )
    # REQ: SYSTEM-004
    add_test(NAME EXAMPLE_CAMERA_IMU_ARTIFACT_TARGET_TEST
        COMMAND ${CMAKE_COMMAND}
            "-DHOLOSCAN_ARTIFACT_BUILD_DIR=${CMAKE_BINARY_DIR}"
            "-DHOLOSCAN_ARTIFACT_TARGET=camera_imu_control_graph_artifacts"
            "-DHOLOSCAN_ARTIFACT_ROOT=${CMAKE_CURRENT_BINARY_DIR}/holoscan-artifacts"
            "-DHOLOSCAN_ARTIFACT_BUILD_CONFIG=$<CONFIG>"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/test_graph_artifacts.cmake"
    )
    set_tests_properties(EXAMPLE_CAMERA_IMU_ARTIFACT_TARGET_TEST
        PROPERTIES
            RUN_SERIAL TRUE
            TIMEOUT 30
    )
endif()

install(TARGETS camera_imu_control
    RUNTIME DESTINATION "${HOLOSCAN_EXAMPLES_INSTALL_DIR}/camera_imu_control/cpp"
    COMPONENT holoscan-examples
)
