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

}  // namespace holoscan::config::test

// Full object codec for Settings: encodes count and name as a two-entry object; decode reads
// them as required fields and calls ObjectDecoder::finish() to release the decode context.
template <>
struct holoscan::config::ConfigCodec<holoscan::config::test::Settings> {
  static ConfigResult<test::Settings> decode(ConfigView view, const DecodeOptions& options,
                                             DecodeContext& context) {
    ObjectDecoder object{std::move(view), options, context};
    auto count = object.required<std::int32_t>("count");
    if (!count) {
      return tl::make_unexpected(count.error());
    }
    auto name = object.required<std::string>("name");
    if (!name) {
      return tl::make_unexpected(name.error());
    }
    auto finished = object.finish();
    if (!finished) {
      return tl::make_unexpected(finished.error());
    }
    return test::Settings{.count = *count, .name = std::move(*name)};
  }

  static ConfigResult<ConfigValue> encode(const test::Settings& value, const EncodeOptions& options,
                                          EncodeContext& context) {
    auto count = ConfigCodec<std::int32_t>::encode(value.count, options, context);
    auto name = ConfigCodec<std::string>::encode(value.name, options, context);
    if (!count) {
      return tl::make_unexpected(count.error());
    }
    if (!name) {
      return tl::make_unexpected(name.error());
    }
    std::array entries{ConfigObjectEntry{"count", *count}, ConfigObjectEntry{"name", *name}};
    return ConfigValue::object(entries);
  }
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
    return ConfigValue::object(entries);
  }
};

namespace holoscan::config::test {
namespace {

static_assert(ConfigCodable<Settings>);
static_assert(ConfigCodable<std::vector<std::int32_t>>);
static_assert(ConfigCodable<std::optional<std::string>>);
static_assert(!ConfigCodable<std::filesystem::path>);
static_assert(ConfigCodable<std::map<std::string, std::int32_t>>);
static_assert(ConfigCodable<std::map<std::string, std::int32_t, std::less<>>>);

TEST(ConfigCodec, ExplicitWholeObjectCodecRoundTripsWithoutSchema) {
  // REQ: CONFIG-006
  const Settings source{.count = 7, .name = "camera"};
  auto encoded = encode_config(source);
  ASSERT_TRUE(encoded);
  auto decoded = decode_config<Settings>(encoded->view());
  ASSERT_TRUE(decoded);
  EXPECT_EQ(*decoded, source);
  auto document_result = Document::from_value(*encoded);
  ASSERT_TRUE(document_result);
  const Document document = std::move(*document_result);
  EXPECT_EQ(document.get<Settings>("").value(), source);
}

TEST(ConfigCodec, StrictNumericPolicyRejectsNarrowingAndCheckedLossyIsExplicit) {
  // REQ: CONFIG-007
  auto floating = ConfigValue::floating(1.25);
  ASSERT_TRUE(floating);
  EXPECT_EQ(decode_config<float>(floating->view()).value(), 1.25F);
  EXPECT_FALSE(decode_config<std::int32_t>(floating->view()));
  auto integral = ConfigValue::floating(3.0);
  ASSERT_TRUE(integral);
  EXPECT_EQ(decode_config<std::int32_t>(integral->view(),
                                        DecodeOptions{MissingValuePolicy::kReject,
                                                      UnknownFieldPolicy::kReject,
                                                      NumericConversionPolicy::kAllowCheckedLossy})
                .value(),
            3);
}

TEST(ConfigCodec, BuiltinContainersPreserveOrderAndDeterministicMapKeys) {
  // REQ: CONFIG-008
  const std::vector<std::int32_t> values{1, 2, 3};
  auto encoded = encode_config(values);
  ASSERT_TRUE(encoded);
  EXPECT_EQ(decode_config<std::vector<std::int32_t>>(encoded->view()).value(), values);

  const std::map<std::string, std::int32_t> mapping{{"b", 2}, {"a", 1}};
  auto map_value = encode_config(mapping);
  ASSERT_TRUE(map_value);
  EXPECT_EQ(decode_config<std::remove_cvref_t<decltype(mapping)>>(map_value->view()).value(),
            mapping);

  const std::map<std::string, std::int32_t, std::less<>> transparent_mapping{{"d", 4}, {"c", 3}};
  auto transparent_value = encode_config(transparent_mapping);
  ASSERT_TRUE(transparent_value);
  EXPECT_EQ(
      decode_config<std::remove_cvref_t<decltype(transparent_mapping)>>(transparent_value->view())
          .value(),
      transparent_mapping);
}

TEST(ConfigCodec, TypedPatchMergesObjectsAndLeavesBaselineUnchangedOnFailure) {
  // REQ: CONFIG-009
  const Settings baseline{.count = 1, .name = "stable"};
  auto count = ConfigValue::signed_integer(4);
  ASSERT_TRUE(count);
  std::array entries{ConfigObjectEntry{"count", *count}};
  auto patch = ConfigValue::object(entries);
  ASSERT_TRUE(patch);
  auto document_result = Document::from_value(*patch);
  ASSERT_TRUE(document_result);
  const Document document = std::move(*document_result);
  auto updated = document.patch(Path::root(), baseline);
  ASSERT_TRUE(updated);
  EXPECT_EQ(updated->count, 4);
  EXPECT_EQ(updated->name, "stable");
  EXPECT_EQ(baseline.count, 1);
}

TEST(ConfigCodec, PackageContextPreservesNestedPathsAndRequiresDecoderFinish) {
  // REQ: CONFIG-006, CONFIG-009
  // Key setup: array whose element is the wrong type to produce a nested decode error.
  auto text = ConfigValue::string("not-an-integer");
  ASSERT_TRUE(text);
  const std::array values{*text};
  auto array = ConfigValue::array(values);
  ASSERT_TRUE(array);
  auto nested_failure = decode_config<std::vector<std::int32_t>>(array->view());
  ASSERT_FALSE(nested_failure);
  // Key assertion: error path carries the child index ("/0") and kTypeMismatch.
  EXPECT_EQ(nested_failure.error().path(), "/0");
  EXPECT_EQ(nested_failure.error().code(), ConfigErrorCode::kTypeMismatch);
  EXPECT_EQ(nested_failure.error().cause_count(), 1U);

  // Key setup: valid object fed to UnfinishedSettings codec which never calls finish().
  auto count = ConfigValue::signed_integer(7);
  ASSERT_TRUE(count);
  const std::array entries{ConfigObjectEntry{"count", *count}};
  auto object = ConfigValue::object(entries);
  ASSERT_TRUE(object);
  auto unfinished = decode_config<UnfinishedSettings>(object->view());
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
  const auto large_integer = ConfigValue::signed_integer(16'777'217);
  ASSERT_TRUE(large_integer);
  EXPECT_FALSE(decode_config<float>(large_integer->view()));
  EXPECT_TRUE(decode_config<float>(large_integer->view(), checked));

  auto fractional = ConfigValue::floating(3.5);
  ASSERT_TRUE(fractional);
  EXPECT_FALSE(decode_config<std::int32_t>(fractional->view(), checked));
  auto underflow = ConfigValue::floating(std::numeric_limits<double>::denorm_min());
  ASSERT_TRUE(underflow);
  EXPECT_FALSE(decode_config<float>(underflow->view(), checked));
  auto maximum = ConfigValue::unsigned_integer(std::numeric_limits<std::uint64_t>::max());
  ASSERT_TRUE(maximum);
  EXPECT_FALSE(decode_config<std::int64_t>(maximum->view()));
}

}  // namespace
}  // namespace holoscan::config::test
