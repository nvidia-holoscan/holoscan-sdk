/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/config/codec.hpp>

namespace holoscan::config::test {

struct NothrowValue {};
struct ThrowingValue {};

}  // namespace holoscan::config::test

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::NothrowValue> {
  static ConfigResult<test::NothrowValue> decode(
      ConfigView,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions&, DecodeContext&) noexcept {
    return test::NothrowValue{};
  }
  static ConfigResult<ConfigValue> encode(const test::NothrowValue&, const EncodeOptions&,
                                          EncodeContext&) noexcept {
    return ConfigValue::null();
  }
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::ThrowingValue> {
  static ConfigResult<test::ThrowingValue> decode(
      ConfigView,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions&, DecodeContext&) {
    return test::ThrowingValue{};
  }
  static ConfigResult<ConfigValue> encode(const test::ThrowingValue&, const EncodeOptions&,
                                          EncodeContext&) {
    return ConfigValue::null();
  }
};

static_assert(holoscan::config::ConfigCodable<holoscan::config::test::NothrowValue>);
static_assert(!holoscan::config::ConfigCodable<holoscan::config::test::ThrowingValue>);

int main() {
  return 0;
}
