/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/config/codec.hpp>
#include <holoscan/config/io.hpp>

#include <tuple>

namespace holoscan::config::test {

struct NothrowValue {};
struct NothrowObject {
  NothrowValue value;
};
struct ThrowingValue {};
struct ThrowingDefaultObject {
  ThrowingDefaultObject() noexcept(false) {}
  NothrowValue value;
};
struct ThrowingAssignmentValue {
  ThrowingAssignmentValue() = default;
  ThrowingAssignmentValue(const ThrowingAssignmentValue&) = default;
  ThrowingAssignmentValue(ThrowingAssignmentValue&&) = default;
  ThrowingAssignmentValue& operator=(const ThrowingAssignmentValue&) = default;
  ThrowingAssignmentValue& operator=(ThrowingAssignmentValue&&) noexcept(false) { return *this; }
};
struct ThrowingAssignmentObject {
  ThrowingAssignmentValue value;
};

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

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::ThrowingAssignmentValue> {
  static ConfigResult<test::ThrowingAssignmentValue> decode(
      ConfigView,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions&, DecodeContext&) noexcept {
    return test::ThrowingAssignmentValue{};
  }
  static ConfigResult<ConfigValue> encode(const test::ThrowingAssignmentValue&,
                                          const EncodeOptions&, EncodeContext&) noexcept {
    return ConfigValue::null();
  }
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::NothrowObject>
    : holoscan::config::ObjectCodec<holoscan::config::test::NothrowObject> {
  static constexpr auto fields = std::tuple{
      object_field("value", &holoscan::config::test::NothrowObject::value),
  };
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::ThrowingDefaultObject>
    : holoscan::config::ObjectCodec<holoscan::config::test::ThrowingDefaultObject> {
  static constexpr auto fields = std::tuple{
      object_field("value", &holoscan::config::test::ThrowingDefaultObject::value),
  };
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::ThrowingAssignmentObject>
    : holoscan::config::ObjectCodec<holoscan::config::test::ThrowingAssignmentObject> {
  static constexpr auto fields = std::tuple{
      object_field("value", &holoscan::config::test::ThrowingAssignmentObject::value),
  };
};

template struct holoscan::config::ObjectCodec<holoscan::config::test::NothrowObject>;

// REQ: CONFIG-006
static_assert(holoscan::config::ConfigCodable<holoscan::config::test::NothrowValue>);
static_assert(!holoscan::config::ConfigCodable<holoscan::config::test::ThrowingValue>);
static_assert(!holoscan::config::ConfigCodable<holoscan::config::test::NothrowObject>);
static_assert(!holoscan::config::ConfigCodable<holoscan::config::test::ThrowingDefaultObject>);
static_assert(!holoscan::config::ConfigCodable<holoscan::config::test::ThrowingAssignmentObject>);

// REQ: CONFIG-020
[[maybe_unused]] void instantiate_throwing_header_forms(
    const holoscan::config::Document& document) {
  using holoscan::config::test::NothrowValue;
  static_cast<void>(holoscan::config::decode_config<NothrowValue>(document.root()));
  static_cast<void>(document.get<NothrowValue>(""));
  static_cast<void>(document.replace("", NothrowValue{}));
}

int main() {
  return 0;
}
