# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Share the actual Core declaration with packaging tests using inert producers.
# This file is included standalone by those tests, so it carries its own default.
if(NOT DEFINED HOLOSCAN_BUILD_PYTHON_STUBS)
    set(HOLOSCAN_BUILD_PYTHON_STUBS ON)
endif()
set(_holoscan_python_files
    __init__.py
    _module_discovery.py
    _public_api.py
    _tools/__init__.py
    _tools/check_stub_coverage.py
    _tools/check_type_completeness.py
    _tools/generate_operator_stubs.py
    channel/__init__.py
    channel/aio.py
    clocks.py
    config.py
    diagnostics.py
    errors.py
    execution.py
    graphs/__init__.py
    lifecycle.py
    operators/__init__.py
    params.py
    payload/__init__.py
    payload/schema.py
    registry.py
    schema/__init__.py
    schema/geometry.py
    schema/header.py
    schema/image.py
    schema/imu.py
    schema/string.py
    schema/tensor_packet.py
    schema/time.py
    sensor_io/__init__.py
    tensor.py
    time.py
    topics.py)
# Ship typing only as a complete set. py.typed without the generated public facade
# stubs would advertise a typed package whose symbols no type checker can resolve.
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _holoscan_python_files
        __init__.pyi
        channel/__init__.pyi
        channel/aio.pyi
        clocks.pyi
        config.pyi
        diagnostics.pyi
        errors.pyi
        execution.pyi
        graphs/__init__.pyi
        lifecycle.pyi
        operators/__init__.pyi
        params.pyi
        payload/__init__.pyi
        payload/schema.pyi
        py.typed
        registry.pyi
        schema/__init__.pyi
        schema/geometry.pyi
        schema/header.pyi
        schema/image.pyi
        schema/imu.pyi
        schema/string.pyi
        schema/tensor_packet.pyi
        schema/time.pyi
        sensor_io/__init__.pyi
        tensor.pyi
        time.pyi
        topics.pyi)
endif()

# Wheel ownership is broader than the ordinary Core Python install component.
# Select built targets, never stale files left by a testing-enabled configuration.
set(_holoscan_wheel_python_files ${_holoscan_python_files})
set(_holoscan_wheel_testing_targets)
if(TARGET holoscan_testing_python)
    list(APPEND _holoscan_wheel_python_files testing/__init__.py)
    if(HOLOSCAN_BUILD_PYTHON_STUBS)
        list(APPEND _holoscan_wheel_python_files testing/__init__.pyi testing/py.typed)
    endif()
    list(APPEND _holoscan_wheel_testing_targets testing)
endif()

_holoscan_get_iceoryx2_ffi_library(_holoscan_iceoryx2_ffi_library)
# Share schema and ImageT runtimes through Core instead of duplicating them in member wheels.
holoscan_add_python_wheel_component(
    MODULE core PACKAGE holoscan
    PYTHON_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}"
    PYTHON_FILES ${_holoscan_wheel_python_files}
    EXTENSION_TARGET holoscan_core_python
    NATIVE_TARGETS
        config
        config_io
        core
        holoscan_schema
        logger
        params
        sensor_io
        ${_holoscan_wheel_testing_targets}
        holomq
        iceoryx2-cxx::shared-lib-cxx
    NATIVE_FILES "${_holoscan_iceoryx2_ffi_library}")
get_property(_holoscan_builtin_schema_targets GLOBAL
    PROPERTY HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS)
foreach(_holoscan_builtin_schema_target IN LISTS _holoscan_builtin_schema_targets)
    get_target_property(_holoscan_builtin_schema_codec_target
        "${_holoscan_builtin_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET)
    if(NOT _holoscan_builtin_schema_codec_target OR
       _holoscan_builtin_schema_codec_target MATCHES "-NOTFOUND$")
        continue()
    endif()
    _holoscan_add_python_wheel_extension(
        holoscan-wheel-core holoscan "${_holoscan_builtin_schema_codec_target}" schema)
endforeach()
unset(_holoscan_builtin_schema_codec_target)
unset(_holoscan_builtin_schema_target)
unset(_holoscan_builtin_schema_targets)
unset(_holoscan_iceoryx2_ffi_library)

if(TARGET holoscan_testing_python)
    _holoscan_add_python_wheel_extension(
        holoscan-wheel-core holoscan holoscan_testing_python testing)
endif()
unset(_holoscan_wheel_python_files)
unset(_holoscan_wheel_testing_targets)
