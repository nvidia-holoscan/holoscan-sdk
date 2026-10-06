/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <string_view>

#include <holoscan/core/schema_identity.hpp>

namespace holoscan::test {

[[nodiscard]] consteval std::byte schema_id_hex_nibble(const char value) {
  if (value >= '0' && value <= '9') {
    return std::byte{static_cast<unsigned char>(value - '0')};
  }
  if (value >= 'a' && value <= 'f') {
    return std::byte{static_cast<unsigned char>(value - 'a' + 10)};
  }
  std::abort();
}

[[nodiscard]] consteval SchemaId schema_id_from_hex(const std::string_view hex) {
  if (hex.size() != 32U) {
    std::abort();
  }
  SchemaId result;
  for (std::size_t index = 0U; index < result.value.size(); ++index) {
    result.value[index] =
        schema_id_hex_nibble(hex[index * 2U]) << 4U | schema_id_hex_nibble(hex[index * 2U + 1U]);
  }
  return result;
}

}  // namespace holoscan::test
