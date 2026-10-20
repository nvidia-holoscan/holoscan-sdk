# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

foreach(required_variable IN ITEMS
        HOLOSCAN_BUILD_DIR
        HOLOSCAN_INSTALL_PREFIX
        HOLOSCAN_INSTALL_LIBDIR
        HOLOSCAN_PYTHON_EXECUTABLE
        HOLOSCAN_PYTHON_VERSION
        HOLOSCAN_BUILD_PYTHON_STUBS)
    if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()
unset(required_variable)

file(REMOVE_RECURSE "${HOLOSCAN_INSTALL_PREFIX}")

function(_holoscan_install_component component)
    set(install_command
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${HOLOSCAN_INSTALL_PREFIX}"
        --component "${component}")
    if(DEFINED HOLOSCAN_BUILD_TYPE AND NOT HOLOSCAN_BUILD_TYPE STREQUAL "")
        list(APPEND install_command --config "${HOLOSCAN_BUILD_TYPE}")
    endif()
    execute_process(COMMAND ${install_command} RESULT_VARIABLE install_result)
    if(NOT install_result EQUAL 0)
        message(FATAL_ERROR "Failed to install SDK component ${component}")
    endif()
endfunction()

foreach(component IN ITEMS
        holoscan-dependencies
        iceoryx2-lib
        holoscan-holomq
        holoscan-core
        holoscan-config
        holoscan-config-io
        holoscan-params
        holoscan-sensor-io
        holoscan-python_libs
        holoscan-visualization)
    _holoscan_install_component("${component}")
endforeach()
unset(component)

string(CONCAT site_packages
    "${HOLOSCAN_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIBDIR}/python${HOLOSCAN_PYTHON_VERSION}"
    "/site-packages")
set(visualization_package "${site_packages}/holoscan_visualization")
if(EXISTS "${visualization_package}")
    message(FATAL_ERROR
        "Native Visualization staging unexpectedly installed ${visualization_package}")
endif()

set(python_environment
    "PYTHONPATH=${site_packages}"
    "LD_LIBRARY_PATH=${HOLOSCAN_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIBDIR}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${python_environment}
        "${HOLOSCAN_PYTHON_EXECUTABLE}" -c
        "import holoscan; import importlib.util; assert importlib.util.find_spec('holoscan_visualization') is None"
    RESULT_VARIABLE native_import_result
    ERROR_VARIABLE native_import_error)
if(NOT native_import_result EQUAL 0)
    message(FATAL_ERROR "Native-only Python import failed:\n${native_import_error}")
endif()

_holoscan_install_component(holoscan-visualization-python_libs)
set(required_paths
        __init__.py
        _visualization.abi3.so
        operators/__init__.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND required_paths
        __init__.pyi
        _visualization.pyi
        operators/__init__.pyi
        py.typed)
endif()
foreach(required_path IN LISTS required_paths)
    if(NOT EXISTS "${visualization_package}/${required_path}")
        message(FATAL_ERROR "Installed Visualization package is missing ${required_path}")
    endif()
endforeach()
unset(required_path)
unset(required_paths)

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${python_environment}
        "${HOLOSCAN_PYTHON_EXECUTABLE}" -c
        "from holoscan_visualization.operators import HolovizOp, ImageLayer, WindowSink; assert HolovizOp.name == 'holoscan.visualization.HolovizOp'; assert ImageLayer(port_name='v', max_width=1, max_height=1).kind == 'image'; assert WindowSink.kind == 'window'"
    RESULT_VARIABLE visualization_import_result
    ERROR_VARIABLE visualization_import_error)
if(NOT visualization_import_result EQUAL 0)
    message(FATAL_ERROR "Installed Visualization Python import failed:\n${visualization_import_error}")
endif()

# The component also installs the distribution metadata that registers the
# holoscan.visualization alias with Core's module discovery.
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env ${python_environment}
        "${HOLOSCAN_PYTHON_EXECUTABLE}" -c
        "import holoscan.visualization, holoscan_visualization; assert holoscan.visualization is holoscan_visualization"
    RESULT_VARIABLE alias_import_result
    ERROR_VARIABLE alias_import_error)
if(NOT alias_import_result EQUAL 0)
    message(FATAL_ERROR "Installed Visualization alias import failed:\n${alias_import_error}")
endif()
