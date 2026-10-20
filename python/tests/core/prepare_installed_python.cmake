# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED HOLOSCAN_BUILD_DIR OR NOT IS_DIRECTORY "${HOLOSCAN_BUILD_DIR}")
    message(FATAL_ERROR "HOLOSCAN_BUILD_DIR must identify the configured SDK build tree")
endif()
if(NOT DEFINED HOLOSCAN_INSTALL_PREFIX)
    message(FATAL_ERROR "HOLOSCAN_INSTALL_PREFIX is required")
endif()
if(NOT DEFINED HOLOSCAN_INSTALL_LIBDIR OR NOT DEFINED HOLOSCAN_PYTHON_VERSION OR
   NOT DEFINED HOLOSCAN_REQUIRE_TYPING)
    message(FATAL_ERROR "The Python install layout is required")
endif()
if(NOT DEFINED HOLOSCAN_INSTALL_SENSOR_IO)
    set(HOLOSCAN_INSTALL_SENSOR_IO ON)
endif()

cmake_path(IS_PREFIX HOLOSCAN_BUILD_DIR "${HOLOSCAN_INSTALL_PREFIX}" NORMALIZE install_is_staged)
if(NOT install_is_staged)
    message(FATAL_ERROR "The Python test install must remain inside the SDK build tree")
endif()

file(REMOVE_RECURSE "${HOLOSCAN_INSTALL_PREFIX}")
execute_process(
    COMMAND
        "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${HOLOSCAN_INSTALL_PREFIX}"
        --component holoscan-python_libs
    RESULT_VARIABLE install_status
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error
)
if(NOT install_status EQUAL 0)
    message(FATAL_ERROR
        "Failed to stage the installed Python package (${install_status})\n"
        "${install_output}\n${install_error}")
endif()

if(HOLOSCAN_INSTALL_SENSOR_IO)
    execute_process(
        COMMAND
            "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
            --prefix "${HOLOSCAN_INSTALL_PREFIX}"
            --component holoscan-sensor-io-python_libs
        RESULT_VARIABLE sensor_io_install_status
        OUTPUT_VARIABLE sensor_io_install_output
        ERROR_VARIABLE sensor_io_install_error
    )
    if(NOT sensor_io_install_status EQUAL 0)
        message(FATAL_ERROR
            "Failed to stage the Sensor I/O Python package (${sensor_io_install_status})\n"
            "${sensor_io_install_output}\n${sensor_io_install_error}")
    endif()
endif()

if(HOLOSCAN_INSTALL_TESTING)
    execute_process(COMMAND "${CMAKE_COMMAND}" --install "${HOLOSCAN_BUILD_DIR}"
        --prefix "${HOLOSCAN_INSTALL_PREFIX}" --component holoscan-testing-python_libs
        RESULT_VARIABLE testing_status OUTPUT_VARIABLE testing_output ERROR_VARIABLE testing_error)
    if(NOT testing_status EQUAL 0)
        message(FATAL_ERROR "Failed to stage optional testing: ${testing_output}\n${testing_error}")
    endif()
endif()

string(CONCAT package_directory
    "${HOLOSCAN_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIBDIR}/python${HOLOSCAN_PYTHON_VERSION}"
    "/site-packages/holoscan")
set(required_paths __init__.py _core.abi3.so)
if(HOLOSCAN_INSTALL_TESTING)
    list(APPEND required_paths testing/__init__.py testing/_testing.abi3.so)
    if(HOLOSCAN_REQUIRE_TYPING)
        list(APPEND required_paths testing/__init__.pyi testing/py.typed)
    endif()
endif()
if(HOLOSCAN_REQUIRE_TYPING)
    list(APPEND required_paths py.typed __init__.pyi)
endif()
foreach(required_path IN LISTS required_paths)
    if(NOT EXISTS "${package_directory}/${required_path}")
        message(FATAL_ERROR "Installed Python package is missing ${required_path}")
    endif()
endforeach()
set(sensor_io_paths
    schema/image.py
    schema/imu.py
    schema/_image_holoscan.abi3.so
    schema/_imu_holoscan.abi3.so
)
if(HOLOSCAN_REQUIRE_TYPING)
    list(APPEND sensor_io_paths schema/image.pyi schema/imu.pyi)
endif()
foreach(sensor_io_path IN LISTS sensor_io_paths)
    if(HOLOSCAN_INSTALL_SENSOR_IO AND NOT EXISTS "${package_directory}/${sensor_io_path}")
        message(FATAL_ERROR "Installed Sensor I/O Python package is missing ${sensor_io_path}")
    elseif(NOT HOLOSCAN_INSTALL_SENSOR_IO AND EXISTS "${package_directory}/${sensor_io_path}")
        message(FATAL_ERROR "Core-only Python package contains Sensor I/O path ${sensor_io_path}")
    endif()
endforeach()
if(HOLOSCAN_REQUIRE_TYPING)
    file(READ "${package_directory}/schema/__init__.pyi" schema_root_stub)
    foreach(sensor_io_type IN ITEMS Image Imu)
        string(FIND "${schema_root_stub}" "import ${sensor_io_type} as ${sensor_io_type}"
            sensor_io_stub_import)
        if(HOLOSCAN_INSTALL_SENSOR_IO AND sensor_io_stub_import EQUAL -1)
            message(FATAL_ERROR
                "Sensor I/O root stub does not export ${sensor_io_type}")
        elseif(NOT HOLOSCAN_INSTALL_SENSOR_IO AND NOT sensor_io_stub_import EQUAL -1)
            message(FATAL_ERROR
                "Core-only root stub advertises absent Sensor I/O type ${sensor_io_type}")
        endif()
    endforeach()
endif()
foreach(legacy_path IN ITEMS
        core/__init__.py
        core/__init__.pyi
        core/_core.abi3.so
        core/_core.pyi)
    if(EXISTS "${package_directory}/${legacy_path}")
        message(FATAL_ERROR "Installed Python package contains legacy ${legacy_path}")
    endif()
endforeach()
