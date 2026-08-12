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
target_include_directories(holoscan_core_python PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(holoscan_core_python PRIVATE holoscan::core)
add_dependencies(holoscan-python holoscan_core_python)
set_target_properties(holoscan_core_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${HOLOSCAN_PYTHON_CORE_BINARY_DIR}"
    OUTPUT_NAME _core
    # _core lives under python/lib/holoscan/core while libholoscan_core.so
    # lives under lib in both the build tree and installed SDK layout.
    BUILD_WITH_INSTALL_RPATH ON
    INSTALL_RPATH "\$ORIGIN/../../../../lib"
)
