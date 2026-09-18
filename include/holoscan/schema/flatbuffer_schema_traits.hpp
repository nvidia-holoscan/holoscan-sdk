/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "holoscan/core/schema_identity.hpp"

namespace holoscan::schema {

/// @brief Bind one generated FlatBuffers root to its owning type and Holoscan policy.
/// @tparam Root Public generated table root.
///
/// One generated specialization is the source of truth for root/native association, stable
/// identity, finite capacities, and descriptor facts. The canonical name comes from FlatBuffers
/// `--gen-name-strings`; application authors do not repeat it.
template <typename Root>
struct FlatBufferSchemaBinding;

/// @brief Check whether a generated root uses ordinary self-contained FlatBuffer carriage.
template <typename Root>
concept OrdinaryFlatBufferSchemaRoot = requires {
  FlatBufferSchemaBinding<Root>::descriptor_tensor;
} && !FlatBufferSchemaBinding<Root>::descriptor_tensor;

/// @brief Stable family identity derived from the generated root name and intentional epoch.
template <typename Root>
inline constexpr SchemaFamilyIdentity flatbuffer_schema_family_identity_v =
    FlatBufferSchemaBinding<Root>::family_identity;

/// @brief Stable compact family ID for a generated FlatBuffer root.
template <typename Root>
inline constexpr SchemaId flatbuffer_schema_family_id_v =
    flatbuffer_schema_family_identity_v<Root>.id;

/// @brief Exact independent in-buffer identifier for a generated root.
template <typename Root>
inline constexpr FlatBufferFileIdentifier flatbuffer_file_identifier_v =
    FlatBufferSchemaBinding<Root>::file_identifier;

/// @brief Null-terminated identifier passed to FlatBuffers generated APIs.
template <typename Root>
inline constexpr std::array<char, 5> flatbuffer_file_identifier_c_string_v =
    flatbuffer_file_identifier_c_string(flatbuffer_file_identifier_v<Root>);

/// @brief Complete stable semantic identity for a generated root.
template <typename Root>
inline constexpr SchemaIdentity flatbuffer_schema_identity_v =
    FlatBufferSchemaBinding<Root>::identity;

/// @brief Return the complete serialized-payload bound for an ordinary generated root.
template <OrdinaryFlatBufferSchemaRoot Root>
inline constexpr std::size_t flatbuffer_schema_max_serialized_size_v =
    FlatBufferSchemaBinding<Root>::max_serialized_size;

/// @brief Fixed metadata-size bound for a descriptor-Tensor generated root.
template <typename Root>
inline constexpr std::size_t flatbuffer_descriptor_metadata_capacity_v =
    FlatBufferSchemaBinding<Root>::metadata_capacity;

}  // namespace holoscan::schema
