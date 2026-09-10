# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_PYTHON_CORE_BINARY_DIR "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/core")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_CORE_BINARY_DIR}")

set(HOLOSCAN_PYTHON_CORE_INIT_OUTPUT "${HOLOSCAN_PYTHON_CORE_BINARY_DIR}/__init__.py")
add_custom_command(
    OUTPUT "${HOLOSCAN_PYTHON_CORE_INIT_OUTPUT}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
            "${HOLOSCAN_PYTHON_CORE_INIT_OUTPUT}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py"
    COMMENT "Copying holoscan.core Python package __init__.py"
)
add_custom_target(holoscan-python-core-init DEPENDS "${HOLOSCAN_PYTHON_CORE_INIT_OUTPUT}")
add_dependencies(holoscan-python holoscan-python-core-init)

nanobind_add_module(holoscan_core_python STABLE_ABI _core.cpp tensor.cpp dl_converter.cpp)
target_include_directories(holoscan_core_python PRIVATE ${holoscan_SOURCE_DIR}/src)
target_link_libraries(holoscan_core_python PRIVATE holoscan::core magic_enum::magic_enum)
add_dependencies(holoscan-python holoscan_core_python)
file(RELATIVE_PATH HOLOSCAN_PYTHON_CORE_INSTALL_LIBDIR_RELATIVE_PATH
    "/${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}"
    "/${CMAKE_INSTALL_LIBDIR}")
set_target_properties(holoscan_core_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOSCAN_PYTHON_CORE_BINARY_DIR}"
    OUTPUT_NAME _core
    # The build-tree module lives under python/lib/holoscan/core while the
    # installed module lives under lib/python<version>/site-packages/holoscan/core.
    # Keep their origin-relative paths separate so build-tree Python tests do not
    # use the installed-layout RPATH.
    BUILD_RPATH "\$ORIGIN/../../../../lib"
    INSTALL_RPATH "\$ORIGIN/${HOLOSCAN_PYTHON_CORE_INSTALL_LIBDIR_RELATIVE_PATH}"
)
