# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_PYTHON_CORE_BINARY_DIR "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}")

option(HOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE
    "Build the core Python module against the external nanobind split-mode backend" OFF)
mark_as_advanced(HOLOSCAN_PYTHON_NANOBIND_SPLIT_MODE_PROTOTYPE)

set(_HOLOSCAN_PYTHON_CORE_SOURCES
    _core.cpp
    channel.cpp
    channel_tensor.cpp
    clocks.cpp
    config.cpp
    dl_converter.cpp
    errors.cpp
    execution_authoring.cpp
    graph.cpp
    lifecycle_authoring.cpp
    native_io.cpp
    native_operator.cpp
    payload.cpp
    plan.cpp
    py_object_payload.cpp
    python_path.cpp
    registry.cpp
    sensor_io.cpp
    schema_value.cpp
    session.cpp
    session_reaper.cpp
    temporal.cpp
    tensor.cpp
    tensor_authoring.cpp
    time.cpp
    topics.cpp
)
file(RELATIVE_PATH HOLOSCAN_PYTHON_CORE_INSTALL_LIBDIR_RELATIVE_PATH
    "/${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}"
    "/${CMAKE_INSTALL_LIBDIR}")

function(_holoscan_configure_core_python_target target output_directory output_name)
    target_link_libraries(${target}
        PRIVATE
            holoscan::core
            holoscan::config_io
            holoscan::params
            holoscan::schema
            magic_enum::magic_enum
    )
    set_target_properties(${target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${output_directory}"
        OUTPUT_NAME "${output_name}"
        # The build-tree module lives under python/lib/holoscan while the
        # installed module lives under lib/python<version>/site-packages/holoscan.
        # Keep their origin-relative paths separate so build-tree Python tests do not
        # use the installed-layout RPATH.
        BUILD_RPATH "\$ORIGIN/../../../lib"
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
    # Generate each public facade independently. Native objects report these public modules through
    # __module__, so stubbing the private flat extension would produce facade imports instead of useful
    # definitions and could introduce a stub cycle.
    set(_HOLOSCAN_PYTHON_PUBLIC_STUBS
        "holoscan|__init__.pyi"
        "holoscan.channel|channel/__init__.pyi"
        "holoscan.clocks|clocks.pyi"
        "holoscan.config|config.pyi"
        "holoscan.diagnostics|diagnostics.pyi"
        "holoscan.errors|errors.pyi"
        "holoscan.execution|execution.pyi"
        "holoscan.lifecycle|lifecycle.pyi"
        "holoscan.payload|payload/__init__.pyi"
        "holoscan.payload.schema|payload/schema.pyi"
        "holoscan.params|params.pyi"
        "holoscan.registry|registry.pyi"
        "holoscan.sensor_io|sensor_io/__init__.pyi"
        "holoscan.tensor|tensor.pyi"
        "holoscan.time|time.pyi"
        "holoscan.topics|topics.pyi"
    )
    set(_HOLOSCAN_PYTHON_GENERATED_PUBLIC_STUBS)
    set(_HOLOSCAN_PYTHON_STAGED_PUBLIC_STUBS)
    foreach(stub_spec IN LISTS _HOLOSCAN_PYTHON_PUBLIC_STUBS)
        string(REPLACE "|" ";" stub_fields "${stub_spec}")
        list(GET stub_fields 0 stub_module)
        list(GET stub_fields 1 stub_path)
        string(REPLACE "." "-" stub_target_suffix "${stub_module}")
        set(raw_stub "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/raw/${stub_path}")
        set(generated_stub "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/${stub_path}")
        set(package_stub "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${stub_path}")

        holoscan_add_nanobind_stubs(
            TARGET holoscan-python-${stub_target_suffix}-raw-stub
            EXTENSION_TARGETS holoscan_core_python
            PYTHON_EXECUTABLE "${Python3_EXECUTABLE}"
            GENERATOR_SCRIPT "${_holoscan_nanobind_stubgen}"
            MODULE "${stub_module}"
            OUTPUT "${raw_stub}"
            PYTHON_PATH "${HOLOSCAN_PYTHON_MODULE_ROOT}"
            DEPENDS
                holoscan_core_python
                holoscan-python-init
                ${HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES}
        )
        add_custom_command(
            OUTPUT "${generated_stub}"
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../../finalize_public_stub.py"
                    --input "${raw_stub}"
                    --output "${generated_stub}"
            DEPENDS
                "${raw_stub}"
                "${CMAKE_CURRENT_SOURCE_DIR}/../../finalize_public_stub.py"
            COMMENT "Finalizing ${stub_module} public stub"
        )
        list(APPEND _HOLOSCAN_PYTHON_GENERATED_PUBLIC_STUBS "${generated_stub}")
        if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../${stub_path}")
            add_custom_command(
                OUTPUT "${package_stub}"
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${generated_stub}"
                        "${package_stub}"
                DEPENDS "${generated_stub}"
                COMMENT "Staging ${stub_module} public stub in the build package"
            )
            list(APPEND _HOLOSCAN_PYTHON_STAGED_PUBLIC_STUBS "${package_stub}")
        endif()
    endforeach()
    add_custom_target(generate-python-facade-stubs
        DEPENDS ${_HOLOSCAN_PYTHON_GENERATED_PUBLIC_STUBS})
    if(_HOLOSCAN_PYTHON_STAGED_PUBLIC_STUBS)
        add_custom_target(holoscan-python-facade-stubs
            DEPENDS ${_HOLOSCAN_PYTHON_STAGED_PUBLIC_STUBS})
        add_dependencies(holoscan-python holoscan-python-facade-stubs)
    endif()
endif()
