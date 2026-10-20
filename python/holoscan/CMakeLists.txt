# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/cmake/setup_dependencies.cmake")
include(HoloscanStubGeneration)
include(HoloscanPythonWheel)

# Use the generator from the SDK's pinned nanobind checkout.
find_file(_holoscan_nanobind_stubgen stubgen.py PATHS "${NB_DIR}/src" "${NB_DIR}"
    NO_DEFAULT_PATH NO_CACHE REQUIRED)

add_custom_target(holoscan-python ALL)
# The Core wheel owns these runtimes even when no member module is enabled.
add_dependencies(holoscan-python holoscan_schema sensor_io)

set(HOLOSCAN_PYTHON_MODULE_BINARY_DIR "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/channel")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/operators")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/graphs")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/payload")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/schema")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/sensor_io")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/_tools")
set(HOLOSCAN_PYTHON_GENERATED_STUB_DIR
    "${holoscan_BINARY_DIR}/python/generated-stubs/holoscan")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/raw/channel")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/raw/payload")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/channel")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/raw/sensor_io")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/graphs")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/payload")
file(MAKE_DIRECTORY "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/sensor_io")

set(HOLOSCAN_PYTHON_LEGACY_CORE_DIR "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/core")
if(EXISTS "${HOLOSCAN_PYTHON_LEGACY_CORE_DIR}")
    file(REMOVE_RECURSE "${HOLOSCAN_PYTHON_LEGACY_CORE_DIR}")
endif()

set(HOLOSCAN_PYTHON_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set(HOLOSCAN_PYTHON_CORE_INSTALL_DIR "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan")

function(_holoscan_copy_python_package_file relative_path)
    set(output "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${relative_path}")
    add_custom_command(
        OUTPUT "${output}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}"
                "${output}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${relative_path}"
        COMMENT "Copying holoscan Python package file ${relative_path}"
    )
    set(HOLOSCAN_PYTHON_PACKAGE_OUTPUTS
        ${HOLOSCAN_PYTHON_PACKAGE_OUTPUTS} "${output}" PARENT_SCOPE)
endfunction()

foreach(package_file IN ITEMS
        __init__.py
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
        payload/__init__.py
        payload/schema.py
        params.py
        registry.py
        schema/__init__.py
        sensor_io/__init__.py
        tensor.py
        time.py
        topics.py
        operators/__init__.py)
    _holoscan_copy_python_package_file("${package_file}")
endforeach()

add_custom_target(holoscan-python-init DEPENDS ${HOLOSCAN_PYTHON_PACKAGE_OUTPUTS})
add_dependencies(holoscan-python holoscan-python-init)

set(_holoscan_python_copied_files _module_discovery.py)
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    list(APPEND _holoscan_python_copied_files
        __init__.pyi
        channel/__init__.pyi
        channel/aio.pyi
        clocks.pyi
        config.pyi
        diagnostics.pyi
        errors.pyi
        execution.pyi
        lifecycle.pyi
        operators/__init__.pyi
        params.pyi
        payload/__init__.pyi
        payload/schema.pyi
        py.typed
        registry.pyi
        schema/__init__.pyi
        sensor_io/__init__.pyi
        tensor.pyi
        time.pyi
        topics.pyi)
endif()
foreach(package_file IN LISTS _holoscan_python_copied_files)
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${package_file}"
        "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${package_file}" COPYONLY)
endforeach()
# The full SDK build tree and Core wheel bundle Sensor I/O. Native component installs override
# this full root stub below so Core-only installs do not advertise absent Image and Imu modules.
if(HOLOSCAN_BUILD_PYTHON_STUBS)
    configure_file("${holoscan_SOURCE_DIR}/src/sensor_io/python/schema_init.pyi"
        "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/schema/__init__.pyi" COPYONLY)
endif()

get_property(HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES GLOBAL
    PROPERTY HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS)
foreach(_holoscan_schema_import_target IN LISTS HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES)
    get_target_property(_holoscan_schema_import_codec_target
        "${_holoscan_schema_import_target}" HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET)
    if(_holoscan_schema_import_codec_target AND
       NOT _holoscan_schema_import_codec_target MATCHES "-NOTFOUND$")
        list(APPEND HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES
            "${_holoscan_schema_import_codec_target}")
    endif()
endforeach()
list(REMOVE_DUPLICATES HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES)
add_dependencies(holoscan-python ${HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES})
set(HOLOSCAN_SDK_STUBGEN_DEPENDS
    holoscan-python-init
    ${HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES})
unset(_holoscan_schema_import_codec_target)
unset(_holoscan_schema_import_target)

# One conservative interface/build identity is shared by the independently packaged bindings.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/BindingServicesCompatibility.cmake")
set(HOLOSCAN_PYTHON_BINDING_SERVICES_INCLUDE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/binding-services")
holoscan_configure_binding_services_compatibility(
    "${holoscan_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_LIST_DIR}/core"
    "${NB_DIR}"
    "${HOLOSCAN_PYTHON_BINDING_SERVICES_INCLUDE_DIR}")

add_subdirectory(core)
# Later sibling modules cannot inherit this directory's temporary dependency variables.
# Keep the import prerequisites with the Core extension they pass to the stub helpers.
set_property(TARGET holoscan_core_python PROPERTY
    HOLOSCAN_PYTHON_IMPORT_DEPENDENCIES ${HOLOSCAN_SDK_STUBGEN_DEPENDS})
if(HOLOSCAN_BUILD_TESTING_LIBRARY)
    add_subdirectory(testing)
endif()

# Vision's native target is configured earlier; its bindings need nanobind and Python first.
if(TARGET holoscan::vision)
    add_subdirectory("${holoscan_SOURCE_DIR}/modules/vision/python"
        "${holoscan_BINARY_DIR}/modules/vision/python")
endif()

# Visualization owns its extension and nested Python package; Core only supplies the common
# registry/handle types used by the provider facade.
if(TARGET holoscan::visualization::op_holoviz)
    add_subdirectory("${holoscan_SOURCE_DIR}/modules/visualization/python"
        "${holoscan_BINARY_DIR}/modules/visualization/python")
endif()

# Inference owns its extension and Python package; Core remains independent of TensorRT and the
# module-specific configuration grammar.
if(TARGET holoscan::op_inference)
    add_subdirectory("${holoscan_SOURCE_DIR}/modules/inference/python"
        "${holoscan_BINARY_DIR}/modules/inference/python")
endif()
unset(HOLOSCAN_SDK_STUBGEN_DEPENDS)
unset(HOLOSCAN_PYTHON_SCHEMA_IMPORT_DEPENDENCIES)

if(HOLOSCAN_BUILD_PYTHON_STUBS)
    # Reusable Graph proxy types are populated dynamically, so their generic public
    # stub is source-owned rather than derived from the process-local registry.
    set(HOLOSCAN_PYTHON_GENERATED_GRAPH_STUB
        "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/graphs/__init__.pyi")
    add_custom_command(
        OUTPUT "${HOLOSCAN_PYTHON_GENERATED_GRAPH_STUB}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${CMAKE_CURRENT_SOURCE_DIR}/graphs/__init__.pyi"
                "${HOLOSCAN_PYTHON_GENERATED_GRAPH_STUB}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/graphs/__init__.pyi"
        COMMENT "Staging the generic reusable Graph stub"
    )
    add_custom_target(generate-python-graph-stubs
        DEPENDS "${HOLOSCAN_PYTHON_GENERATED_GRAPH_STUB}")
    add_custom_command(
        OUTPUT "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/graphs/__init__.pyi"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${HOLOSCAN_PYTHON_GENERATED_GRAPH_STUB}"
                "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/graphs/__init__.pyi"
        DEPENDS "${HOLOSCAN_PYTHON_GENERATED_GRAPH_STUB}"
        COMMENT "Staging the generic reusable Graph stub in the build package"
    )
    add_custom_target(holoscan-python-graph-stubs
        DEPENDS "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/graphs/__init__.pyi")
    add_dependencies(holoscan-python holoscan-python-graph-stubs)

    add_custom_target(generate-python-stubs
        DEPENDS
            generate-python-facade-stubs
            generate-python-graph-stubs
    )
    if(TARGET generate-holoscan-vision-python-stubs)
        add_dependencies(generate-python-stubs generate-holoscan-vision-python-stubs)
    endif()

    set(_HOLOSCAN_PYTHON_COMMITTED_STUBS
        __init__.pyi
        channel/__init__.pyi
        clocks.pyi
        config.pyi
        diagnostics.pyi
        errors.pyi
        execution.pyi
        lifecycle.pyi
        params.pyi
        payload/__init__.pyi
        payload/schema.pyi
        registry.pyi
        sensor_io/__init__.pyi
        tensor.pyi
        time.pyi
        topics.pyi
    )
    set(_HOLOSCAN_PYTHON_STUB_CHECK_COMMANDS)
    foreach(stub_path IN LISTS _HOLOSCAN_PYTHON_COMMITTED_STUBS)
        list(APPEND _HOLOSCAN_PYTHON_STUB_CHECK_COMMANDS
            COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/_tools/check_stub_coverage.py"
                    --generated "${HOLOSCAN_PYTHON_GENERATED_STUB_DIR}/${stub_path}"
                    --committed "${CMAKE_CURRENT_SOURCE_DIR}/${stub_path}"
        )
    endforeach()
    add_custom_target(check-python-stubs
        ${_HOLOSCAN_PYTHON_STUB_CHECK_COMMANDS}
        COMMAND ${CMAKE_COMMAND} -E compare_files
                "${CMAKE_CURRENT_SOURCE_DIR}/graphs/__init__.pyi"
                "${HOLOSCAN_PYTHON_GENERATED_GRAPH_STUB}"
        DEPENDS generate-python-stubs
                "${CMAKE_CURRENT_SOURCE_DIR}/_tools/check_stub_coverage.py"
        COMMENT "Checking committed Python stubs"
    )
    if(TARGET check-holoscan-vision-python-stubs)
        add_dependencies(check-python-stubs check-holoscan-vision-python-stubs)
    endif()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/cmake/wheel.cmake")
# wheel.cmake stays self-contained for the packaging tests, so its payload list
# is declared separately from the copy list above. Fail closed if they drift.
foreach(copied IN LISTS _holoscan_python_copied_files)
    if(NOT copied IN_LIST _holoscan_python_files)
        message(FATAL_ERROR
            "'${copied}' is copied into the build tree but missing from "
            "_holoscan_python_files in cmake/wheel.cmake; the wheel would omit it")
    endif()
endforeach()
get_property(_holoscan_builtin_schema_targets GLOBAL
    PROPERTY HOLOSCAN_PYTHON_SCHEMA_INSTALL_TARGETS)
set(_holoscan_native_python_excluded_files)
foreach(_holoscan_builtin_schema_target IN LISTS _holoscan_builtin_schema_targets)
    get_target_property(_holoscan_builtin_schema_component
        "${_holoscan_builtin_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_INSTALL_COMPONENT)
    if(_holoscan_builtin_schema_component STREQUAL "holoscan-python_libs")
        continue()
    endif()
    get_target_property(_holoscan_builtin_schema_module
        "${_holoscan_builtin_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_MODULE)
    file(RELATIVE_PATH _holoscan_builtin_schema_module
        "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}" "${_holoscan_builtin_schema_module}")
    list(APPEND _holoscan_native_python_excluded_files
        "${_holoscan_builtin_schema_module}")
    if(HOLOSCAN_BUILD_PYTHON_STUBS)
        get_target_property(_holoscan_builtin_schema_stub
            "${_holoscan_builtin_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_STUB)
        file(RELATIVE_PATH _holoscan_builtin_schema_stub
            "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}" "${_holoscan_builtin_schema_stub}")
        list(APPEND _holoscan_native_python_excluded_files
            "${_holoscan_builtin_schema_stub}")
    endif()
endforeach()
foreach(path IN LISTS _holoscan_python_files)
    if(path IN_LIST _holoscan_native_python_excluded_files)
        continue()
    endif()
    get_filename_component(directory "${path}" DIRECTORY)
    if(path STREQUAL "schema/__init__.pyi")
        install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/${path}"
            DESTINATION "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan/${directory}"
            COMPONENT holoscan-python_libs)
    else()
        install(FILES "${HOLOSCAN_PYTHON_MODULE_BINARY_DIR}/${path}"
            DESTINATION "${HOLOSCAN_PYTHON_INSTALL_DIR}/holoscan/${directory}"
            COMPONENT holoscan-python_libs)
    endif()
endforeach()
install(TARGETS holoscan_core_python
    LIBRARY DESTINATION "${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}" COMPONENT holoscan-python_libs)
foreach(_holoscan_builtin_schema_target IN LISTS _holoscan_builtin_schema_targets)
    get_target_property(_holoscan_builtin_schema_component
        "${_holoscan_builtin_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_INSTALL_COMPONENT)
    if(NOT _holoscan_builtin_schema_component STREQUAL "holoscan-python_libs")
        continue()
    endif()
    get_target_property(_holoscan_builtin_schema_codec_target
        "${_holoscan_builtin_schema_target}" HOLOSCAN_FLATBUFFER_PYTHON_CODEC_TARGET)
    if(NOT _holoscan_builtin_schema_codec_target OR
       _holoscan_builtin_schema_codec_target MATCHES "-NOTFOUND$")
        continue()
    endif()
    install(TARGETS "${_holoscan_builtin_schema_codec_target}"
        LIBRARY DESTINATION "${HOLOSCAN_PYTHON_CORE_INSTALL_DIR}/schema"
        COMPONENT holoscan-python_libs)
endforeach()
unset(_holoscan_builtin_schema_codec_target)
unset(_holoscan_builtin_schema_component)
unset(_holoscan_builtin_schema_module)
unset(_holoscan_builtin_schema_stub)
unset(_holoscan_builtin_schema_target)
unset(_holoscan_builtin_schema_targets)
unset(_holoscan_native_python_excluded_files)
