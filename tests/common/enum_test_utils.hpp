/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <bit>
#include <type_traits>

namespace holoscan::test {

template <typename Enum>
  requires std::is_enum_v<Enum>
[[nodiscard]] constexpr Enum enum_from_underlying(std::underlying_type_t<Enum> value) noexcept {
  return std::bit_cast<Enum>(value);
}

}  // namespace holoscan::test
