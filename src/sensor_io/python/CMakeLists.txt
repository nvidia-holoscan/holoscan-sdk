# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Image and Imu extend the Core-owned holoscan.schema package. Native SDK installs keep their
# Python files in the separately selectable Sensor I/O component, while the Core Python wheel
# deliberately bundles that component's generated values, codecs, and shared library.
set(_sensor_io_python_package "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan")
set(_sensor_io_python_install
    "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/holoscan")
file(RELATIVE_PATH _sensor_io_python_build_lib_relative
    "${_sensor_io_python_package}/schema"
    "${holoscan_BINARY_DIR}/lib")
file(RELATIVE_PATH _sensor_io_python_install_lib_relative
    "/${_sensor_io_python_install}/schema"
    "/${CMAKE_INSTALL_LIBDIR}")
get_property(_sensor_io_schema_targets GLOBAL
    PROPERTY HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS)
foreach(_sensor_io_schema_target IN LISTS _sensor_io_schema_targets)
    get_target_property(_sensor_io_python_component
        "${_sensor_io_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_INSTALL_COMPONENT)
    if(NOT _sensor_io_python_component STREQUAL "holoscan-sensor-io-python_libs")
        continue()
    endif()
    get_target_property(_sensor_io_python_module
        "${_sensor_io_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_MODULE)
    get_target_property(_sensor_io_python_stub
        "${_sensor_io_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_STUB)
    get_target_property(_sensor_io_python_codec
        "${_sensor_io_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET)

    set_target_properties("${_sensor_io_python_codec}" PROPERTIES
        BUILD_RPATH "\$ORIGIN/${_sensor_io_python_build_lib_relative}"
        INSTALL_RPATH "\$ORIGIN/${_sensor_io_python_install_lib_relative}")

    get_filename_component(_sensor_io_python_module_name "${_sensor_io_python_module}" NAME)
    install(FILES "${_sensor_io_python_module}"
        DESTINATION "${_sensor_io_python_install}/schema"
        COMPONENT holoscan-sensor-io-python_libs)
    if(HOLOSCAN_BUILD_PYTHON_STUBS)
        get_filename_component(_sensor_io_python_stub_name "${_sensor_io_python_stub}" NAME)
        install(FILES "${_sensor_io_python_stub}"
            DESTINATION "${_sensor_io_python_install}/schema"
            COMPONENT holoscan-sensor-io-python_libs)
    endif()
    install(TARGETS "${_sensor_io_python_codec}"
        LIBRARY DESTINATION "${_sensor_io_python_install}/schema"
        COMPONENT holoscan-sensor-io-python_libs)
endforeach()
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    # Overlay Core's root stub when the optional Sensor I/O component is selected. COPY_FILE is
    # intentional: install(FILES) can skip the distinct overlay when both source stubs have the
    # same checkout timestamp.
    install(CODE "
        file(MAKE_DIRECTORY
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_sensor_io_python_install}/schema\")
        file(COPY_FILE
            \"${CMAKE_CURRENT_SOURCE_DIR}/schema_init.pyi\"
            \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${_sensor_io_python_install}/schema/__init__.pyi\"
            ONLY_IF_DIFFERENT)
        "
        COMPONENT holoscan-sensor-io-python_libs)
endif()

unset(_sensor_io_python_codec)
unset(_sensor_io_python_component)
unset(_sensor_io_python_build_lib_relative)
unset(_sensor_io_python_install)
unset(_sensor_io_python_install_lib_relative)
unset(_sensor_io_python_module)
unset(_sensor_io_python_module_name)
unset(_sensor_io_python_package)
unset(_sensor_io_python_stub)
unset(_sensor_io_python_stub_name)
unset(_sensor_io_schema_target)
unset(_sensor_io_schema_targets)
