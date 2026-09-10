# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")

add_custom_target(holoscan-python ALL)

set(HOLOSCAN_PYTHON_MODULE_BINARY_DIR "${holoscan_BINARY_DIR}/python/lib/holoscan")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}")

set(HOLOSCAN_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set(HOLOSCAN_PYTHON_CORE_INSTALL_DIR "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan/core")

set(HOLOSCAN_PYTHON_INIT_OUTPUT "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/__init__.py")
add_custom_command(
    OUTPUT "${HOLOSCAN_PYTHON_INIT_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
            "${HOLOSCAN_PYTHON_INIT_OUTPUT}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    COMMENT "Copying holoscan Python package __init__.py"
)
add_custom_target(holoscan-python-init DEPENDS "${HOLOSCAN_PYTHON_INIT_OUTPUT}")
add_dependencies(holoscan-python holoscan-python-init)

add_subdirectory(core)

install(
    DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}"
    DESTINATION "${HOLOSCAN_PYTHON_INSTALL_DIR}"
    COMPONENT holoscan-python_libs
    PATTERN "__pycache__" EXCLUDE
)
