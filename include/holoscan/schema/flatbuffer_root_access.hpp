/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <flatbuffers/flatbuffers.h>

#include "holoscan/schema/flatbuffer_schema_traits.hpp"

namespace holoscan::schema::detail {

/// @brief Verify nonempty bytes as one exact generated FlatBuffers root.
/// @tparam Root Generated FlatBuffers root type.
/// @param bytes Nonempty serialized root bytes.
/// @return Whether the bytes pass the generated verifier and exact file-identifier check.
template <typename Root>
[[nodiscard]] inline bool verify_flatbuffer_root_bytes(
    const std::span<const std::byte> bytes) noexcept {
  flatbuffers::Verifier verifier{reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()};
  return verifier.VerifyBuffer<Root>(flatbuffer_file_identifier_c_string_v<Root>.data());
}

/// @brief Parse one generated root from bytes that have already been verified.
/// @tparam Root Generated FlatBuffers root type.
/// @param bytes Verified serialized root bytes, or an empty span.
/// @return Borrowed generated root, or nullptr for an empty span.
template <typename Root>
[[nodiscard]] inline const Root* parse_flatbuffer_root_bytes(
    const std::span<const std::byte> bytes) noexcept {
  if (bytes.empty()) {
    return nullptr;
  }
  return flatbuffers::GetRoot<Root>(reinterpret_cast<const std::uint8_t*>(bytes.data()));
}

}  // namespace holoscan::schema::detail
