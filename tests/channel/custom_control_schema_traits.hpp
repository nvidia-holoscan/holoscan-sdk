/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include <holoscan/core/schema_identity.hpp>
#include <holoscan/schema/flatbuffer_schema_traits.hpp>

#include "custom_control_generated.h"

// Shared trait specialization for the test-only application-defined FlatBuffer root
// `holoscan::tests::channel::CustomControl` (CHAN-031). It is included by both the in-process
// schema test (`schema.cpp`) and the standalone interprocess child
// (`schema_interprocess_child.cpp`) so the two processes agree on one hand-pinned identity; a
// cross-process schema channel would otherwise fail closed on any drift between separately
// maintained copies of the digest/id.

namespace holoscan::schema {

template <>
struct FlatBufferSchemaTraits<::holoscan::tests::channel::CustomControl> {
  using root_type = ::holoscan::tests::channel::CustomControl;
  static constexpr SchemaIdentity identity{
      .canonical_name = "holoscan.tests.channel.CustomControl",
      .version = 1U,
      .schema_sha256 = detail::schema_hex_bytes<32>(
          "d7be43402b6e42d44fe6d1b5098202457ae0bd5e1be7ede9f8211906a2da4d76"),
      .id = SchemaId{.value = detail::schema_hex_bytes<16>("0ca574685c9db19bf1ed421b9aa5079d")},
  };
  static constexpr std::uint32_t schema_version = 1U;
  static constexpr std::string_view root_name = identity.canonical_name;
  static constexpr std::size_t default_inline_capacity = 512U;
  static constexpr std::size_t default_descriptor_metadata_capacity = 512U;
};

}  // namespace holoscan::schema

namespace holoscan {

template <>
consteval SchemaIdentity schema_identity<tests::channel::CustomControl>() {
  return schema::FlatBufferSchemaTraits<tests::channel::CustomControl>::identity;
}

}  // namespace holoscan
