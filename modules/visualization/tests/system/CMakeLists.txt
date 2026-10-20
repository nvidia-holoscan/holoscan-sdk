# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${holoscan_SOURCE_DIR}/tests/cmake/consumer_test.cmake")

find_program(HOLOSCAN_DUMPBIN_EXECUTABLE NAMES dumpbin)

# REQ: SYSTEM-016
# REQ: SYSTEM-017
# REQ: COMPILE-059
holoscan_add_installed_consumer_test(
    VISUALIZATION.installed_package_consumer
    SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/operator_installed_consumer"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/operator_installed_consumer"
    RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/run_operator_installed_consumer.cmake"
    LABELS visualization package-boundary
    CMAKE_ARGS
        "-DHOLOSCAN_NM_EXECUTABLE=${CMAKE_NM}"
        "-DHOLOSCAN_DUMPBIN_EXECUTABLE=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
        "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
        "-DHOLOSCAN_CONSUMER_FIND_MODE=component"
)

holoscan_add_installed_consumer_test(
    VISUALIZATION.installed_module_package_consumer
    SOURCE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/operator_installed_consumer"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/operator_installed_module_consumer"
    RUNNER "${CMAKE_CURRENT_SOURCE_DIR}/run_operator_installed_consumer.cmake"
    LABELS visualization package-boundary
    CMAKE_ARGS
        "-DHOLOSCAN_NM_EXECUTABLE=${CMAKE_NM}"
        "-DHOLOSCAN_DUMPBIN_EXECUTABLE=${HOLOSCAN_DUMPBIN_EXECUTABLE}"
        "-DHOLOSCAN_EXPECT_SHARED=$<BOOL:${BUILD_SHARED_LIBS}>"
        "-DHOLOSCAN_CONSUMER_FIND_MODE=module"
        "-DHOLOSCAN_MODULE_PACKAGE_DIR=${CMAKE_INSTALL_LIBDIR}/cmake/holoscan"
)
