# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Generate the owning object-API type and the Holoscan package traits used by
# Input<app::CustomControlT> and Output<app::CustomControlT>.
set(CUSTOM_SCHEMA_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
holoscan_add_flatbuffer_schema(
    TARGET custom_schema_flatbuffers_generated
    SCHEMAS custom_control.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${CUSTOM_SCHEMA_GENERATED_DIR}"
    BFBS_EMBED
    TRAITS_HEADER custom_control_schema_traits.hpp
    TRAITS_SCHEMA custom_control.fbs
    TRAITS_ROOT_TYPE app::CustomControl
    TRAITS_NATIVE_TYPE app::CustomControlT
    TRAITS_BINARY_SCHEMA_TYPE app::CustomControlBinarySchema
    TRAITS_PACKAGE_NAMESPACE app::package
    TRAITS_CANONICAL_NAME app.CustomControl
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 512
)
