/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/config/codec.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace holoscan::config::test {

// Round-trip test fixture: a two-field aggregate with an explicit ConfigCodec specialization
// covering both encode and decode paths.  Tests confirm the codec produces the same value
// after a full encode → decode cycle.
// Representative test: ExplicitWholeObjectCodecRoundTripsWithoutSchema.
struct Settings {
  std::int32_t count{0};
  std::string name;
  friend bool operator==(const Settings&, const Settings&) = default;
};

// Intentionally incomplete decoder: ConfigCodec::decode returns without calling
// ObjectDecoder::finish(), simulating a codec that forgets the finalization step.
// Used to verify the package boundary enforces finish() even when the caller appears
// to succeed.  Representative test: PackageContextPreservesNestedPathsAndRequiresDecoderFinish.
struct UnfinishedSettings {
  std::int32_t count{0};
};

struct ThrowingConfigOperation {};

struct EmptySettings {};

struct DuplicateRequiredSettings {
  std::int32_t first{0};
  std::int32_t second{0};
};

struct DuplicateDefaultedSettings {
  std::int32_t first{0};
  std::int32_t second{0};
};

struct MoveOnlyValue {
  MoveOnlyValue() = default;
  explicit MoveOnlyValue(std::int32_t initial_value) : value{initial_value} {}
  MoveOnlyValue(const MoveOnlyValue&) = delete;
  MoveOnlyValue& operator=(const MoveOnlyValue&) = delete;
  explicit MoveOnlyValue(MoveOnlyValue&& other) noexcept : value{other.value} {}
  MoveOnlyValue& operator=(MoveOnlyValue&&) = default;

  std::int32_t value{0};
};

struct MoveOnlySettings {
  MoveOnlySettings() = default;
  explicit MoveOnlySettings(MoveOnlyValue initial_value) : value{std::move(initial_value)} {}
  MoveOnlySettings(const MoveOnlySettings&) = delete;
  explicit MoveOnlySettings(MoveOnlySettings&& other) noexcept : value{std::move(other.value)} {}

  MoveOnlyValue value;
};

struct NullMemberSettings {
  std::int32_t value{0};
};

struct AssignmentOnlyValue {
  AssignmentOnlyValue() = default;
  AssignmentOnlyValue(const AssignmentOnlyValue&) = delete;
  AssignmentOnlyValue(AssignmentOnlyValue&&) = delete;
  AssignmentOnlyValue& operator=(const AssignmentOnlyValue&) = delete;
  AssignmentOnlyValue& operator=(AssignmentOnlyValue&&) noexcept { return *this; }
};

struct NonMovableMemberSettings {
  AssignmentOnlyValue value;
};

struct NonMovableOwnerSettings {
  NonMovableOwnerSettings() = default;
  NonMovableOwnerSettings(const NonMovableOwnerSettings&) = delete;
  NonMovableOwnerSettings(NonMovableOwnerSettings&&) = delete;

  std::int32_t value{0};
};

}  // namespace holoscan::config::test

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::Settings>
    : holoscan::config::ObjectCodec<holoscan::config::test::Settings> {
  static constexpr auto fields = std::tuple{
      object_field("count", &test::Settings::count),
      defaulted_object_field("name", &test::Settings::name),
  };
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::EmptySettings>
    : holoscan::config::ObjectCodec<holoscan::config::test::EmptySettings> {
  static constexpr auto fields = std::tuple{};
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::DuplicateRequiredSettings>
    : holoscan::config::ObjectCodec<holoscan::config::test::DuplicateRequiredSettings> {
  static constexpr auto fields = std::tuple{
      object_field("value", &test::DuplicateRequiredSettings::first),
      object_field("value", &test::DuplicateRequiredSettings::second),
  };
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::DuplicateDefaultedSettings>
    : holoscan::config::ObjectCodec<holoscan::config::test::DuplicateDefaultedSettings> {
  static constexpr auto fields = std::tuple{
      object_field("value", &test::DuplicateDefaultedSettings::first),
      defaulted_object_field("value", &test::DuplicateDefaultedSettings::second),
  };
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::MoveOnlyValue> {
  static ConfigResult<test::MoveOnlyValue> decode(ConfigView view, const DecodeOptions& options,
                                                  DecodeContext& context) {
    auto decoded = ConfigCodec<std::int32_t>::decode(std::move(view), options, context);
    if (!decoded) {
      return tl::make_unexpected(decoded.error());
    }
    return ConfigResult<test::MoveOnlyValue>{test::MoveOnlyValue{*decoded}};
  }

  static ConfigResult<ConfigValue> encode(const test::MoveOnlyValue& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ConfigCodec<std::int32_t>::encode(value.value, options, context);
  }
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::MoveOnlySettings>
    : holoscan::config::ObjectCodec<holoscan::config::test::MoveOnlySettings> {
  static constexpr auto fields = std::tuple{
      object_field("value", &test::MoveOnlySettings::value),
  };
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::NullMemberSettings>
    : holoscan::config::ObjectCodec<holoscan::config::test::NullMemberSettings> {
  static constexpr auto fields = std::tuple{
      object_field("value", static_cast<std::int32_t test::NullMemberSettings::*>(nullptr))};
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::AssignmentOnlyValue> {
  static ConfigResult<test::AssignmentOnlyValue> decode(
      ConfigView,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions&, DecodeContext&) {
    return ConfigResult<test::AssignmentOnlyValue>{tl::in_place};
  }

  static ConfigResult<ConfigValue> encode(const test::AssignmentOnlyValue&, const EncodeOptions&,
                                          EncodeContext&) {
    return ConfigValue::null();
  }
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::NonMovableMemberSettings>
    : holoscan::config::ObjectCodec<holoscan::config::test::NonMovableMemberSettings> {
  static constexpr auto fields = std::tuple{
      object_field("value", &test::NonMovableMemberSettings::value),
  };
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::NonMovableOwnerSettings>
    : holoscan::config::ObjectCodec<holoscan::config::test::NonMovableOwnerSettings> {
  static constexpr auto fields = std::tuple{
      object_field("value", &test::NonMovableOwnerSettings::value),
  };
};

// Broken codec for UnfinishedSettings: deliberately omits ObjectDecoder::finish() so the
// test can observe the kCodecFailure error surfaced at the package boundary.
template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::UnfinishedSettings> {
  static ConfigResult<test::UnfinishedSettings> decode(ConfigView view,
                                                       const DecodeOptions& options,
                                                       DecodeContext& context) {
    ObjectDecoder object{std::move(view), options, context};
    auto count = object.required<std::int32_t>("count");
    if (!count) {
      return tl::make_unexpected(count.error());
    }
    // Deliberately omit finish(); the package boundary must reject apparent success.
    return test::UnfinishedSettings{.count = *count};
  }

  static ConfigResult<ConfigValue> encode(const test::UnfinishedSettings& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    auto count = context.encode_child<std::int32_t>("count", value.count, options);
    if (!count) {
      return tl::make_unexpected(count.error());
    }
    const std::array entries{ConfigObjectEntry{"count", *count}};
    return ConfigValue::try_object(entries);
  }
};

template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::ThrowingConfigOperation> {
  // Match the owning-view codec contract even though this failure fixture only reads the view.
  static ConfigResult<test::ThrowingConfigOperation> decode(
      ConfigView view,  // NOLINT(performance-unnecessary-value-param)
      const DecodeOptions&, DecodeContext&) {
    static_cast<void>(view.as_string());
    return test::ThrowingConfigOperation{};
  }

  static ConfigResult<ConfigValue> encode(const test::ThrowingConfigOperation&,
                                          const EncodeOptions&, EncodeContext&) {
    const std::string invalid_utf8{"\xC0\x80", 2};
    return ConfigValue::string(invalid_utf8);
  }
};

namespace holoscan::config::test {
namespace {

// REQ: CONFIG-006
// REQ: CONFIG-021
static_assert(ConfigCodable<Settings>);
static_assert(ConfigCodable<EmptySettings>);
static_assert(ConfigCodable<MoveOnlyValue>);
static_assert(ConfigCodable<MoveOnlySettings>);
static_assert(!std::copy_constructible<MoveOnlySettings>);
static_assert(!std::is_convertible_v<MoveOnlyValue&&, MoveOnlyValue>);
static_assert(!std::is_convertible_v<MoveOnlySettings&&, MoveOnlySettings>);
static_assert(ConfigCodable<AssignmentOnlyValue>);
static_assert(!ConfigCodable<DuplicateRequiredSettings>);
static_assert(!ConfigCodable<DuplicateDefaultedSettings>);
static_assert(!ConfigCodable<NullMemberSettings>);
static_assert(!ConfigCodable<NonMovableMemberSettings>);
static_assert(!ConfigCodable<NonMovableOwnerSettings>);
static_assert(ConfigCodable<std::vector<std::int32_t>>);
static_assert(ConfigCodable<std::optional<std::string>>);
static_assert(!ConfigCodable<std::filesystem::path>);
static_assert(ConfigCodable<std::map<std::string, std::int32_t>>);
static_assert(ConfigCodable<std::map<std::string, std::int32_t, std::less<>>>);

TEST(ConfigCodec, ThrowingCompleteOperationsPreserveCodecConfigErrors) {
  // REQ: CONFIG-020
  // REQ: CONFIG-021
  const ConfigValue boolean = ConfigValue::boolean(true);
  auto decode_failure = try_decode_config<ThrowingConfigOperation>(boolean.view());
  ASSERT_FALSE(decode_failure);
  EXPECT_EQ(decode_failure.error().code(), ConfigErrorCode::kTypeMismatch);
  try {
    static_cast<void>(decode_config<ThrowingConfigOperation>(boolean.view()));
    FAIL() << "expected codec decode rejection";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kTypeMismatch);
  }

  auto encode_failure = try_encode_config(ThrowingConfigOperation{});
  ASSERT_FALSE(encode_failure);
  EXPECT_EQ(encode_failure.error().code(), ConfigErrorCode::kInvalidUtf8);
  try {
    static_cast<void>(encode_config(ThrowingConfigOperation{}));
    FAIL() << "expected codec encode rejection";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kInvalidUtf8);
  }
}

TEST(ConfigCodec, ExplicitWholeObjectCodecRoundTripsWithoutSchema) {
  // REQ: CONFIG-006
  const Settings source{.count = 7, .name = "camera"};
  const ConfigValue encoded = encode_config(source);
  EXPECT_EQ(decode_config<Settings>(encoded.view()), source);
  const Document document = Document::from_value(encoded);
  EXPECT_EQ(document.get<Settings>(""), source);
  const Settings replacement{.count = 8, .name = "replacement"};
  EXPECT_EQ(document.replace("", replacement).get<Settings>(""), replacement);

  auto count = ConfigValue::try_signed_integer(3);
  ASSERT_TRUE(count);
  const std::array entries{ConfigObjectEntry{"count", *count}};
  auto defaults = ConfigValue::try_object(entries);
  ASSERT_TRUE(defaults);
  EXPECT_EQ(try_decode_config<Settings>(defaults->view()).value(),
            (Settings{.count = 3, .name = ""}));

  const std::array<ConfigObjectEntry, 0> no_entries{};
  auto missing_required = ConfigValue::try_object(no_entries);
  ASSERT_TRUE(missing_required);
  auto failure = try_decode_config<Settings>(missing_required->view());
  ASSERT_FALSE(failure);
  EXPECT_EQ(failure.error().path(), "/count");

  auto encoded_defaults = try_encode_config(Settings{.count = 3});
  ASSERT_TRUE(encoded_defaults);
  EXPECT_EQ(encoded_defaults->view().size(), 2U);
  EXPECT_EQ(encoded_defaults->view().try_key(1).value(), "name");

  auto encoded_empty = try_encode_config(EmptySettings{});
  ASSERT_TRUE(encoded_empty);
  EXPECT_EQ(encoded_empty->kind(), ConfigValueKind::kObject);
  EXPECT_EQ(encoded_empty->view().size(), 0U);
  EXPECT_TRUE(try_decode_config<EmptySettings>(encoded_empty->view()));

  const MoveOnlySettings move_only_source{MoveOnlyValue{11}};
  auto encoded_move_only = try_encode_config(move_only_source);
  ASSERT_TRUE(encoded_move_only);
  auto decoded_move_only = try_decode_config<MoveOnlySettings>(encoded_move_only->view());
  ASSERT_TRUE(decoded_move_only);
  EXPECT_EQ(decoded_move_only->value.value, 11);

  auto throwing_decoded_move_only = decode_config<MoveOnlySettings>(encoded_move_only->view());
  EXPECT_EQ(throwing_decoded_move_only.value.value, 11);
  const Document move_only_document = Document::from_value(*encoded_move_only);
  auto throwing_get_move_only = move_only_document.get<MoveOnlySettings>(Path::root());
  EXPECT_EQ(throwing_get_move_only.value.value, 11);
  auto throwing_patch_move_only =
      move_only_document.patch<MoveOnlySettings>(Path::root(), MoveOnlySettings{MoveOnlyValue{7}});
  EXPECT_EQ(throwing_patch_move_only.value.value, 11);

  auto scalar = ConfigValue::try_signed_integer(3);
  ASSERT_TRUE(scalar);
  auto empty_object_failure = try_decode_config<EmptySettings>(scalar->view());
  ASSERT_FALSE(empty_object_failure);
  EXPECT_EQ(empty_object_failure.error().code(), ConfigErrorCode::kTypeMismatch);
}

TEST(ConfigCodec, StrictNumericPolicyRejectsNarrowingAndCheckedLossyIsExplicit) {
  // REQ: CONFIG-007
  auto floating = ConfigValue::try_floating(1.25);
  ASSERT_TRUE(floating);
  EXPECT_EQ(try_decode_config<float>(floating->view()).value(), 1.25F);
  EXPECT_FALSE(try_decode_config<std::int32_t>(floating->view()));
  auto integral = ConfigValue::try_floating(3.0);
  ASSERT_TRUE(integral);
  EXPECT_EQ(
      try_decode_config<std::int32_t>(integral->view(),
                                      DecodeOptions{MissingValuePolicy::kReject,
                                                    UnknownFieldPolicy::kReject,
                                                    NumericConversionPolicy::kAllowCheckedLossy})
          .value(),
      3);
}

TEST(ConfigCodec, BuiltinContainersPreserveOrderAndDeterministicMapKeys) {
  // REQ: CONFIG-008
  const std::vector<std::int32_t> values{1, 2, 3};
  auto encoded = try_encode_config(values);
  ASSERT_TRUE(encoded);
  EXPECT_EQ(try_decode_config<std::vector<std::int32_t>>(encoded->view()).value(), values);

  const std::map<std::string, std::int32_t> mapping{{"b", 2}, {"a", 1}};
  auto map_value = try_encode_config(mapping);
  ASSERT_TRUE(map_value);
  const auto decoded_mapping =
      try_decode_config<std::remove_cvref_t<decltype(mapping)>>(map_value->view());
  EXPECT_EQ(decoded_mapping.value(), mapping);

  const std::map<std::string, std::int32_t, std::less<>> transparent_mapping{{"d", 4}, {"c", 3}};
  auto transparent_value = try_encode_config(transparent_mapping);
  ASSERT_TRUE(transparent_value);
  EXPECT_EQ(try_decode_config<std::remove_cvref_t<decltype(transparent_mapping)>>(
                transparent_value->view())
                .value(),
            transparent_mapping);
}

TEST(ConfigCodec, TypedPatchMergesObjectsAndLeavesBaselineUnchangedOnFailure) {
  // REQ: CONFIG-009
  const Settings baseline{.count = 1, .name = "stable"};
  auto count = ConfigValue::try_signed_integer(4);
  ASSERT_TRUE(count);
  std::array entries{ConfigObjectEntry{"count", *count}};
  auto patch = ConfigValue::try_object(entries);
  ASSERT_TRUE(patch);
  auto document_result = Document::try_from_value(*patch);
  ASSERT_TRUE(document_result);
  const Document document = std::move(*document_result);
  const auto updated = document.patch(Path::root(), baseline);
  EXPECT_EQ(updated.count, 4);
  EXPECT_EQ(updated.name, "stable");

  const Document invalid_patch = Document::from_value(ConfigValue::boolean(true));
  auto rejected = invalid_patch.try_patch(Path::root(), baseline);
  ASSERT_FALSE(rejected);
  EXPECT_EQ(rejected.error().code(), ConfigErrorCode::kTypeMismatch);
  EXPECT_EQ(baseline.count, 1);
}

TEST(ConfigCodec, PackageContextPreservesNestedPathsAndRequiresDecoderFinish) {
  // REQ: CONFIG-006, CONFIG-009
  // Key setup: array whose element is the wrong type to produce a nested decode error.
  auto text = ConfigValue::try_string("not-an-integer");
  ASSERT_TRUE(text);
  const std::array values{*text};
  auto array = ConfigValue::try_array(values);
  ASSERT_TRUE(array);
  auto nested_failure = try_decode_config<std::vector<std::int32_t>>(array->view());
  ASSERT_FALSE(nested_failure);
  // Key assertion: error path carries the child index ("/0") and kTypeMismatch.
  EXPECT_EQ(nested_failure.error().path(), "/0");
  EXPECT_EQ(nested_failure.error().code(), ConfigErrorCode::kTypeMismatch);
  EXPECT_EQ(nested_failure.error().cause_count(), 1U);

  // Key setup: valid object fed to UnfinishedSettings codec which never calls finish().
  auto count = ConfigValue::try_signed_integer(7);
  ASSERT_TRUE(count);
  const std::array entries{ConfigObjectEntry{"count", *count}};
  auto object = ConfigValue::try_object(entries);
  ASSERT_TRUE(object);
  auto unfinished = try_decode_config<UnfinishedSettings>(object->view());
  ASSERT_FALSE(unfinished);
  // Key assertion: missing finish() surfaces as kCodecFailure at the package boundary.
  EXPECT_EQ(unfinished.error().code(), ConfigErrorCode::kCodecFailure);
}

TEST(ConfigCodec, NumericMatrixRequiresLosslessDefaultsAndClosedCheckedLossyMode) {
  // REQ: CONFIG-007
  // Key setup: values near representable boundaries for integral↔float conversions.
  const auto checked = DecodeOptions{MissingValuePolicy::kReject,
                                     UnknownFieldPolicy::kReject,
                                     NumericConversionPolicy::kAllowCheckedLossy};
  // Key assertion: default policy rejects any narrowing; checked-lossy mode accepts
  //               representable conversions but still rejects fractional and underflow.
  const auto large_integer = ConfigValue::try_signed_integer(16'777'217);
  ASSERT_TRUE(large_integer);
  EXPECT_FALSE(try_decode_config<float>(large_integer->view()));
  EXPECT_TRUE(try_decode_config<float>(large_integer->view(), checked));

  for (const std::uint64_t exact : {0ULL, 16'777'216ULL, 16'777'218ULL, 1ULL << 63U}) {
    const auto value = ConfigValue::try_unsigned_integer(exact);
    ASSERT_TRUE(value);
    EXPECT_TRUE(try_decode_config<float>(value->view()));
  }

  auto fractional = ConfigValue::try_floating(3.5);
  ASSERT_TRUE(fractional);
  EXPECT_FALSE(try_decode_config<std::int32_t>(fractional->view(), checked));
  auto underflow = ConfigValue::try_floating(std::numeric_limits<double>::denorm_min());
  ASSERT_TRUE(underflow);
  EXPECT_FALSE(try_decode_config<float>(underflow->view(), checked));
  auto maximum = ConfigValue::try_unsigned_integer(std::numeric_limits<std::uint64_t>::max());
  ASSERT_TRUE(maximum);
  EXPECT_FALSE(try_decode_config<std::int64_t>(maximum->view()));
  EXPECT_FALSE(try_decode_config<float>(maximum->view()));
}

}  // namespace
}  // namespace holoscan::config::test
