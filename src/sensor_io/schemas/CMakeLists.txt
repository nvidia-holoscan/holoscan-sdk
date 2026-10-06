# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The wire form is a target of its own, but holoscan::sensor_io links it, so a consumer that adds
# the library gets the schemas without naming a second dependency. One dependency per library is
# the packaging model, and an opt-in target that nothing linked meant the codegen never ran at all.

set(SENSOR_IO_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SENSOR_IO_SCHEMA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(_SENSOR_IO_SCHEMA_PYTHON_OUTPUT_DIR
    "${HOLOSCAN_PYTHON_MODULE_ROOT}/holoscan/schema")
set(_SENSOR_IO_IMAGE_PYTHON_ARGS)
set(_SENSOR_IO_IMU_PYTHON_ARGS)
if(HOLOSCAN_BUILD_PYTHON)
    set(_SENSOR_IO_SCHEMA_PYTHON_ARGS
        PYTHON_INSTALL_COMPONENT holoscan-sensor-io-python_libs
        PYTHON_OUTPUT_DIR "${_SENSOR_IO_SCHEMA_PYTHON_OUTPUT_DIR}")
    list(APPEND _SENSOR_IO_IMAGE_PYTHON_ARGS
        PYTHON
        PYTHON_CODEC_DEFERRED
        ${_SENSOR_IO_SCHEMA_PYTHON_ARGS}
        PYTHON_CONSTANTS
            "UNKNOWN_IMAGE_SIGNIFICANT_BITS=255"
        PYTHON_EXTERNAL_TYPES
            "holoscan.schema.Header=holoscan.schema.header"
        PYTHON_LINK_LIBRARIES sensor_io
        PYTHON_TRAITS_INCLUDE holoscan/sensor_io/sensor_schema_package.hpp)
    list(APPEND _SENSOR_IO_IMU_PYTHON_ARGS
        PYTHON
        ${_SENSOR_IO_SCHEMA_PYTHON_ARGS}
        PYTHON_EXTERNAL_TYPES
            "holoscan.schema.Header=holoscan.schema.header"
            "holoscan.schema.Matrix3f=holoscan.schema.geometry"
            "holoscan.schema.Quaternionf=holoscan.schema.geometry"
            "holoscan.schema.Vector3f=holoscan.schema.geometry")
endif()

# The sensor payload types. They are domain payloads rather than transport primitives: a runtime
# core has no business knowing what a Bayer pattern or an angular velocity covariance is, whereas
# tensor.fbs, time.fbs, header.fbs, and string.fbs are meaningful without naming a sensor and stay
# in Core. Owning these here is what lets a lidar or radar payload be added later without amending
# Core's public contract each time.
#
# Exact BFBS bytes are diagnostic provenance, not compatibility identity. Generate directly from
# the component-owned sources; the helper resolves SDK includes without a caller-supplied root.
#
# Both payloads embed their reflection bytes, matching every other SDK-owned schema that generates
# a binding (see public/schemas/CMakeLists.txt for the rationale).
holoscan_add_flatbuffer_schema(
    TARGET sensor_io_image_schema_generated
    SCHEMA image.fbs
    OUTPUT_DIR "${SENSOR_IO_SCHEMA_GENERATED_DIR}"
    TRAITS_HEADER holoscan/sensor_io/schemas/image_schema_traits.hpp
    MAX_SERIALIZED_SIZE 512
    DESCRIPTOR_TENSOR_CUSTOM_CODEC
    BFBS_EMBED
    ${_SENSOR_IO_IMAGE_PYTHON_ARGS}
)
holoscan_add_flatbuffer_schema(
    TARGET sensor_io_imu_schema_generated
    SCHEMA imu.fbs
    OUTPUT_DIR "${SENSOR_IO_SCHEMA_GENERATED_DIR}"
    TRAITS_HEADER holoscan/sensor_io/schemas/imu_schema_traits.hpp
    MAX_SERIALIZED_SIZE 512
    BFBS_EMBED
    ${_SENSOR_IO_IMU_PYTHON_ARGS}
)
unset(_SENSOR_IO_IMAGE_PYTHON_ARGS)
unset(_SENSOR_IO_IMU_PYTHON_ARGS)
unset(_SENSOR_IO_SCHEMA_PYTHON_ARGS)
unset(_SENSOR_IO_SCHEMA_PYTHON_OUTPUT_DIR)
add_custom_target(sensor_io_sensor_schemas_generated)
add_dependencies(sensor_io_sensor_schemas_generated
    sensor_io_image_schema_generated
    sensor_io_imu_schema_generated
)
set(sensor_io_sensor_schemas_generated_GENERATED_HEADERS)
foreach(schema_target IN ITEMS sensor_io_image_schema_generated sensor_io_imu_schema_generated)
    get_target_property(schema_headers
        "${schema_target}" HOLOSCAN_FLATBUFFER_GENERATED_HEADERS)
    list(APPEND sensor_io_sensor_schemas_generated_GENERATED_HEADERS ${schema_headers})
endforeach()

add_library(sensor_io_schemas INTERFACE)
add_library(holoscan::sensor_io_schemas ALIAS sensor_io_schemas)
add_dependencies(sensor_io_schemas sensor_io_sensor_schemas_generated)

# The generated headers are installed unconditionally, so the codegen has to run in a configuration
# that has no consumer linking this target at all -- tests and examples off, for instance. An
# interface library is not itself built, and a custom target is outside the default build unless it
# says otherwise, so without this the install below names files that were never generated and fails.
set_target_properties(sensor_io_sensor_schemas_generated PROPERTIES EXCLUDE_FROM_ALL FALSE)
target_include_directories(sensor_io_schemas
    INTERFACE
        $<BUILD_INTERFACE:${SENSOR_IO_SCHEMA_GENERATED_DIR}>
        $<INSTALL_INTERFACE:include>
        $<INSTALL_INTERFACE:include/holoscan/sensor_io/schemas>
)
# holoscan::schema rather than holoscan::core, because image_generated.h includes the generated
# header.fbs and tensor.fbs code and only holoscan::schema puts Core's generated directory on the
# include path. The direction is still one-way -- sensor_io to schema to core -- so Core remains
# unaware of this library. A consumer therefore reaches Image and Imu through the single
# holoscan::sensor_io dependency.
target_link_libraries(sensor_io_schemas
    INTERFACE
        flatbuffers::flatbuffers
        ${HOLOSCAN_PACKAGE_NAME}::schema
)

# The generated headers have to be installed beside the target that advertises them, or the
# INSTALL_INTERFACE include directory above names a directory the package never ships. The .fbs
# goes too: a vendor schema that includes this one needs the source to compile against, not just
# the C++ that was generated from it.
install(FILES
    ${sensor_io_sensor_schemas_generated_GENERATED_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/holoscan/sensor_io/schemas
    COMPONENT holoscan-sensor-io
)
install(FILES
    "${SENSOR_IO_SCHEMA_DIR}/image.fbs"
    "${SENSOR_IO_SCHEMA_DIR}/imu.fbs"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/holoscan/schemas/sensor_io
    COMPONENT holoscan-sensor-io
)
