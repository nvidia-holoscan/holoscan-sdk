# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_PYTHON_CORE_BINARY_DIR "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/core")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_CORE_BINARY_DIR}")

option(HOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE
    "Build the core Python module against the external nanobind split-mode backend" OFF)
mark_as_advanced(HOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE)

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

set(_HOLOSCAN_PYTHON_CORE_SOURCES
    _core.cpp
    dl_converter.cpp
    tensor.cpp
)
file(RELATIVE_PATH HOLOSCAN_PYTHON_CORE_INSTALL_LIBDIR_RELATIVE_PATH
    "/${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}"
    "/${CMAKE_INSTALL_LIBDIR}")

function(_holoscan_configure_core_python_target target output_directory output_name)
    target_link_libraries(${target} PRIVATE holoscan::core magic_enum::magic_enum)
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${output_directory}"
        OUTPUT_NAME "${output_name}"
        # The build-tree module lives under python/lib/holoscan/core while the
        # installed module lives under lib/python<version>/site-packages/holoscan/core.
        # Keep their origin-relative paths separate so build-tree Python tests do not
        # use the installed-layout RPATH.
        BUILD_RPATH "\$ORIGIN/../../../../lib"
        INSTALL_RPATH "\$ORIGIN/${HOLOSCAN_PYTHON_CORE_INSTALL_LIBDIR_RELATIVE_PATH}"
    )
endfunction()

if(HOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE)
    nanobind_add_module(holoscan_core_python
        BACKEND_MODULE nanobind_backend
        STABLE_ABI_VERSION 3.12
        ${_HOLOSCAN_PYTHON_CORE_SOURCES}
    )
else()
    nanobind_add_module(holoscan_core_python STABLE_ABI ${_HOLOSCAN_PYTHON_CORE_SOURCES})
endif()
_holoscan_configure_core_python_target(
    holoscan_core_python "${HOLOSCAN_PYTHON_CORE_BINARY_DIR}" _core)
add_dependencies(holoscan-python holoscan_core_python)

if(HOLOSCAN_BUILD_PYTHON_STUBS)
    holoscan_add_nanobind_stubs(
        TARGET holoscan-python-core-stub
        ALL
        EXTENSION_TARGETS holoscan_core_python
        PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
        PYTHON_PATH "${HOLOSCAN_PYTHON_MODULE_ROOT}"
        GENERATOR_SCRIPT "${_holoscan_nanobind_stubgen}"
        MODULE holoscan.core._core
        OUTPUT "${HOLOSCAN_PYTHON_CORE_BINARY_DIR}/_core.pyi"
        DEPENDS holoscan_core_python holoscan-python-init holoscan-python-core-init
    )
    add_dependencies(holoscan-python holoscan-python-core-stub)
endif()
