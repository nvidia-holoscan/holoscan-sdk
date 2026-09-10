# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The wire form is a target of its own, but holoscan::sensor_io links it, so a consumer that adds
# the library gets the schemas without naming a second dependency. One dependency per library is
# the packaging model, and an opt-in target that nothing linked meant the codegen never ran at all.

set(SENSOR_IO_SCHEMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SENSOR_IO_SCHEMA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")

# The sensor payload types. They are domain payloads rather than transport primitives: a runtime
# core has no business knowing what a Bayer pattern or an angular velocity covariance is, whereas
# tensor.fbs, time.fbs, header.fbs, and string.fbs are meaningful without naming a sensor and stay
# in Core. Owning these here is what lets a lidar or radar payload be added later without amending
# Core's public contract each time.
#
# Codegen runs against a staging directory holding these two schemas beside the Core schemas they
# include, and that copy is load-bearing rather than convenience.
#
# flatc records every declaration file inside the .bfbs, spelled relative to the root schema's own
# directory, and the identity digests these schemas already shipped with are taken over exactly those
# bytes. Compiling image.fbs here while reaching header.fbs through an -I pointing at Core's
# directory records "//../../../schemas/header.fbs" instead of "//header.fbs". That changes the
# digest, which changes the schema id, which changes the wire identity of Image -- invalidating every
# recorded payload, while every static check still passes and nothing fails until a consumer rejects
# a message at run time. Staging the includes flat keeps each one recorded as "//<name>.fbs", so
# moving these files out of Core is a source-tree change and not a wire-format change.
#
# tests/sensor_io/test_schema_identity.cpp asserts the resulting digests against the constants in
# sensor_schema_traits.hpp, so removing this staging fails a test rather than silently reissuing the
# schemas under new identities.
set(SENSOR_IO_SCHEMA_STAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/schema_stage")
foreach(_sensor_io_schema image.fbs imu.fbs)
    configure_file("${SENSOR_IO_SCHEMA_DIR}/${_sensor_io_schema}"
        "${SENSOR_IO_SCHEMA_STAGE_DIR}/${_sensor_io_schema}" COPYONLY
    )
endforeach()
# The transitive closure of what image.fbs and imu.fbs include, all Core-owned. configure_file adds
# each to the configure dependencies, so editing one in Core restages it here.
foreach(_core_schema header.fbs time.fbs tensor.fbs geometry.fbs)
    configure_file("${HOLOSCAN_SCHEMA_DIR}/${_core_schema}"
        "${SENSOR_IO_SCHEMA_STAGE_DIR}/${_core_schema}" COPYONLY
    )
endforeach()
unset(_sensor_io_schema)
unset(_core_schema)

holoscan_add_flatbuffer_schema(
    TARGET sensor_io_sensor_schemas_generated
    SCHEMAS "${SENSOR_IO_SCHEMA_STAGE_DIR}/image.fbs" "${SENSOR_IO_SCHEMA_STAGE_DIR}/imu.fbs"
    INCLUDE_DIRS "${SENSOR_IO_SCHEMA_STAGE_DIR}"
    OUTPUT_DIR "${SENSOR_IO_SCHEMA_GENERATED_DIR}"
    BFBS_EMBED
)

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
    DESTINATION include/holoscan/sensor_io/schemas
    COMPONENT holoscan-sensor-io
)
install(FILES
    "${SENSOR_IO_SCHEMA_DIR}/image.fbs"
    "${SENSOR_IO_SCHEMA_DIR}/imu.fbs"
    DESTINATION ${CMAKE_INSTALL_DATADIR}/holoscan/schemas/sensor_io
    COMPONENT holoscan-sensor-io
)
