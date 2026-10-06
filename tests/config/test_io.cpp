/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/config/codec.hpp>
#include <holoscan/config/io.hpp>

#include "config/config_internal.hpp"
#include "config/reflect_backend.hpp"

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace holoscan::config_io::test {
namespace {

constexpr std::string_view kSyntheticApplicationConfig =
    R"yaml(# Synthetic application configuration for YAML structure coverage.
application:
  enabled: true
  endpoint:
    host: "example.invalid"
    port: 4242
  frame:
    width: 320
    height: 200
    planes: 4
  views:
    - label: ""
      style: primary
      opacity: 0.75
      priority: 2

_anchors:
  ordered_channels: &ordered_channels
    # Stable synthetic ordering used by independent alias sites.
    - channel_alpha
    - channel_beta
    - channel_gamma
    - channel_delta
    - channel_epsilon

source:
  channels: *ordered_channels

processor:
  resource: fixtures/example/resource.bin
  channels: *ordered_channels
  stages:
    - source: stream_alpha
      destination: result_alpha
      note:
      width: 2
      samples: 6
    - source: stream_beta
      destination: result_beta
      width: 2
      samples: 6
    - source: stream_gamma
      destination: result_gamma
      width: 2
      samples: 6
    - source: stream_delta
      destination: result_delta
      width: 8
      samples: 6
    - source: stream_epsilon
      destination: result_epsilon
      width: 8
      samples: 6
    - source: stream_zeta
      destination: result_zeta
      width: 8
      samples: 6
    - source: stream_eta
      destination: result_eta
      width: 8

startup:
  channels: *ordered_channels # alias-use inline comment
  settings: fixtures/example/startup.yaml
  timeout: 12.5
)yaml";

[[nodiscard]] std::string error_excerpt(const config::ConfigError& error) {
  const auto bytes = error.excerpt();
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}

TEST(ConfigIo, ThrowingOperationsPreserveStructuredFailures) {
  // REQ: CONFIG-020
  auto parse_failure = try_parse("[", Format::kJson);
  ASSERT_FALSE(parse_failure);
  try {
    static_cast<void>(parse("[", Format::kJson));
    FAIL() << "malformed input did not throw";
  } catch (const config::ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), parse_failure.error().code());
    EXPECT_EQ(exception.error().operation_stage(), parse_failure.error().operation_stage());
  }

  auto document = try_parse("{}", Format::kJson);
  ASSERT_TRUE(document);
  auto save_failure = try_save(*document, {}, Format::kJson);
  ASSERT_FALSE(save_failure);
  EXPECT_EQ(save_failure.error().operation_stage(), config::ConfigOperationStage::kValidate);
  EXPECT_EQ(save_failure.error().destination_state(), config::ConfigDestinationState::kUnchanged);
  try {
    static_cast<void>(save(*document, {}, Format::kJson));
    FAIL() << "invalid destination did not throw";
  } catch (const config::ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), save_failure.error().code());
    EXPECT_EQ(exception.error().operation_stage(), save_failure.error().operation_stage());
    EXPECT_EQ(exception.error().destination_state(), save_failure.error().destination_state());
  }
}

TEST(ConfigIo, JsonParserIsStrictTypedAndCanonical) {
  // REQ: CONFIG-010
  // Key setup: envelope-typed input distinguishes u64/i64/f64; the canonical serialized
  //            form is then cross-checked against reflect-cpp as a differential validator.
  auto document =
      try_parse(R"({"u":{"$holoscan.config.scalar/v1":{"kind":"u64","value":"1"}},"s":1,"f":1.0})",
                Format::kJson);
  ASSERT_TRUE(document);
  EXPECT_EQ(document->try_at("/u")->kind(), config::ConfigValueKind::kUnsignedInteger);
  EXPECT_EQ(document->try_at("/s")->kind(), config::ConfigValueKind::kSignedInteger);
  EXPECT_EQ(document->try_at("/f")->kind(), config::ConfigValueKind::kFloating);
  auto encoded = try_serialize(*document, Format::kJson);
  ASSERT_TRUE(encoded);
  EXPECT_EQ(
      *encoded,
      R"({"f":{"$holoscan.config.scalar/v1":{"kind":"f64","value":"1"}},"s":{"$holoscan.config.scalar/v1":{"kind":"i64","value":"1"}},"u":{"$holoscan.config.scalar/v1":{"kind":"u64","value":"1"}}})");
  EXPECT_EQ(try_parse(R"({"x":1,"x":2})", Format::kJson).error().code(),
            config::ConfigErrorCode::kDuplicateKey);
  EXPECT_EQ(try_parse("[01]", Format::kJson).error().code(), config::ConfigErrorCode::kParseError);

  // Key assertion: reflect-cpp accepts the canonical output and rejects malformed bytes;
  // reflect-cpp remains a private differential check over Holoscan-owned canonical bytes. It is
  // deliberately absent from the production parse path and cannot redefine public semantics.
  EXPECT_TRUE(detail::verify_reflect_generic(*encoded));
  EXPECT_FALSE(detail::verify_reflect_generic("{"));
}

TEST(ConfigIo, JsonRejectsDecodedDuplicateKeysAndMalformedSourceBytes) {
  // REQ: CONFIG-010
  auto duplicate = try_parse(R"({"plain":0,"\u0070lain":1})", Format::kJson);
  ASSERT_FALSE(duplicate);
  EXPECT_EQ(duplicate.error().code(), config::ConfigErrorCode::kDuplicateKey);
  EXPECT_EQ(duplicate.error().operation_stage(), config::ConfigOperationStage::kParse);

  constexpr std::array<std::string_view, 8> malformed{
      R"({"value":1} true)",
      R"([1,])",
      R"("\uD800")",
      R"("\uDC00")",
      R"("\uD800\u0041")",
      R"("\x")",
      R"(01)",
      R"(NaN)",
  };
  for (const auto source : malformed) {
    auto parsed = try_parse(source, Format::kJson);
    ASSERT_FALSE(parsed) << source;
    EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kParseError) << source;
    EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse) << source;
  }

  std::string invalid_utf8{"\""};
  invalid_utf8.push_back(static_cast<char>(0xC3));
  invalid_utf8.push_back('(');
  invalid_utf8.push_back('"');
  auto invalid = try_parse(invalid_utf8, Format::kJson);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code(), config::ConfigErrorCode::kParseError);
  EXPECT_EQ(invalid.error().operation_stage(), config::ConfigOperationStage::kParse);

  auto surrogate_pair = try_parse(R"("\uD83D\uDE00")", Format::kJson);
  ASSERT_TRUE(surrogate_pair);
  auto decoded = surrogate_pair->root().try_as_string();
  ASSERT_TRUE(decoded);
  EXPECT_EQ(*decoded, "\xF0\x9F\x98\x80");
}

TEST(ConfigIo, JsonRawNumbersPreserveKindsAndRejectUnrepresentableValues) {
  // REQ: CONFIG-010
  const auto expect_signed = [](std::string_view source, std::int64_t expected) {
    auto document = try_parse(source, Format::kJson);
    ASSERT_TRUE(document) << source;
    EXPECT_EQ(document->root().kind(), config::ConfigValueKind::kSignedInteger) << source;
    auto value = document->root().try_as_signed_integer();
    ASSERT_TRUE(value) << source;
    EXPECT_EQ(*value, expected) << source;
  };
  const auto expect_unsigned = [](std::string_view source, std::uint64_t expected) {
    auto document = try_parse(source, Format::kJson);
    ASSERT_TRUE(document) << source;
    EXPECT_EQ(document->root().kind(), config::ConfigValueKind::kUnsignedInteger) << source;
    auto value = document->root().try_as_unsigned_integer();
    ASSERT_TRUE(value) << source;
    EXPECT_EQ(*value, expected) << source;
  };

  expect_signed("0", 0);
  expect_signed("-9223372036854775808", std::numeric_limits<std::int64_t>::min());
  expect_signed("9223372036854775807", std::numeric_limits<std::int64_t>::max());
  expect_unsigned("9223372036854775808",
                  static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1U);
  expect_unsigned("18446744073709551615", std::numeric_limits<std::uint64_t>::max());

  auto floating = try_parse("-0.0", Format::kJson);
  ASSERT_TRUE(floating);
  EXPECT_EQ(floating->root().kind(), config::ConfigValueKind::kFloating);
  auto normalized = floating->root().try_as_floating();
  ASSERT_TRUE(normalized);
  EXPECT_EQ(*normalized, 0.0);
  EXPECT_FALSE(std::signbit(*normalized));

  for (const std::string_view source : {"-9223372036854775809", "18446744073709551616", "1e309"}) {
    auto parsed = try_parse(source, Format::kJson);
    ASSERT_FALSE(parsed) << source;
    EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kOutOfRange) << source;
    EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse) << source;
  }
}

TEST(ConfigIo, JsonLosslessSpansUseDecodedPathsAndWholeScalarEnvelopes) {
  // REQ: CONFIG-010, CONFIG-013
  // Key setup: source contains a JSON-escaped key ("a/b~c") and a typed scalar
  //            envelope; the lossless round-trip must address nodes via decoded JSON
  //            Pointer paths while preserving unedited source bytes.
  constexpr std::string_view source =
      R"json({ "\u0061/b~c" : 1, "typed" : )json"
      R"json({ "$holoscan.config.scalar/v1" : { "kind" : "u64", )json"
      R"json("value" : "18446744073709551615" } }, "tail" : "unchanged" })json";
  auto document =
      try_parse(source, Format::kJson, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);
  // Key assertion: at() uses RFC 6901 decoded paths; replace() preserves surrounding
  //               source bytes for unchanged fields including "tail".
  auto decoded_key_value = document->try_at("/a~1b~0c");
  ASSERT_TRUE(decoded_key_value);
  EXPECT_EQ(decoded_key_value->try_as_signed_integer().value(), 1);
  auto envelope_value = document->try_at("/typed");
  ASSERT_TRUE(envelope_value);
  EXPECT_EQ(envelope_value->try_as_unsigned_integer().value(),
            std::numeric_limits<std::uint64_t>::max());

  auto decoded_key = document->try_replace("/a~1b~0c", std::int32_t{27});
  ASSERT_TRUE(decoded_key);
  auto envelope = decoded_key->try_replace("/typed", std::uint64_t{7});
  ASSERT_TRUE(envelope);
  auto encoded = try_serialize(*envelope, Format::kJson);
  ASSERT_TRUE(encoded);
  EXPECT_EQ(*encoded,
            R"json({ "\u0061/b~c" : 27, "typed" : )json"
            R"json({"$holoscan.config.scalar/v1":{"kind":"u64","value":"7"}}, )json"
            R"json("tail" : "unchanged" })json");
}

TEST(ConfigIo, JsonHonorsEveryCallerSuppliedLowerParseLimit) {
  // REQ: CONFIG-011
  ParseOptions source_limit;
  source_limit.max_source_bytes = 2;
  EXPECT_EQ(try_parse("null", Format::kJson, source_limit).error().code(),
            config::ConfigErrorCode::kLimitExceeded);

  ParseOptions depth_limit;
  depth_limit.max_depth = 2;
  EXPECT_TRUE(try_parse(R"({"values":[]})", Format::kJson, depth_limit));
  EXPECT_EQ(try_parse(R"({"values":[0]})", Format::kJson, depth_limit).error().code(),
            config::ConfigErrorCode::kLimitExceeded);

  ParseOptions node_limit;
  node_limit.max_nodes = 2;
  EXPECT_TRUE(try_parse("[0]", Format::kJson, node_limit));
  EXPECT_EQ(try_parse("[0,1]", Format::kJson, node_limit).error().code(),
            config::ConfigErrorCode::kLimitExceeded);

  ParseOptions string_limit;
  string_limit.max_string_bytes = 1;
  EXPECT_TRUE(try_parse(R"("a")", Format::kJson, string_limit));
  EXPECT_EQ(try_parse(R"("\u00e9")", Format::kJson, string_limit).error().code(),
            config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(try_parse(R"({"ab":0})", Format::kJson, string_limit).error().code(),
            config::ConfigErrorCode::kLimitExceeded);

  ParseOptions container_limit;
  container_limit.max_container_entries = 1;
  EXPECT_TRUE(try_parse("[0]", Format::kJson, container_limit));
  EXPECT_EQ(try_parse("[0,1]", Format::kJson, container_limit).error().code(),
            config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(try_parse(R"({"a":0,"b":1})", Format::kJson, container_limit).error().code(),
            config::ConfigErrorCode::kLimitExceeded);
}

TEST(ConfigIo, ReservedLiteralObjectEnvelopeIsBackendNeutral) {
  // REQ: CONFIG-010, CONFIG-018
  constexpr std::string_view source =
      R"({"$holoscan.config.object/v1":[["$holoscan.config.scalar/v1","value"]]})";

  auto json = try_parse(source, Format::kJson);
  auto yaml = try_parse(source, Format::kYaml);
  ASSERT_TRUE(json);
  ASSERT_TRUE(yaml);
  EXPECT_EQ(json->root().kind(), config::ConfigValueKind::kObject);
  EXPECT_EQ(json->root().size(), 1U);
  EXPECT_EQ(json->try_at("/$holoscan.config.scalar~1v1")->try_as_string().value(), "value");
  EXPECT_EQ(json->digest().semantic, yaml->digest().semantic);

  auto json_encoded = try_serialize(*json, Format::kJson);
  auto yaml_encoded = try_serialize(*yaml, Format::kYaml);
  ASSERT_TRUE(json_encoded);
  ASSERT_TRUE(yaml_encoded);
  EXPECT_EQ(*json_encoded, *yaml_encoded);

  constexpr std::string_view malformed =
      R"({"outer":{"$holoscan.config.object/v1":[["missing-value"]]}})";
  for (const auto format : {Format::kJson, Format::kYaml}) {
    const auto parsed = try_parse(malformed, format);
    ASSERT_FALSE(parsed);
    EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kUnsupportedValue);
    EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse);
    if (format == Format::kYaml) {
      EXPECT_EQ(parsed.error().path(), "/outer");
      EXPECT_FALSE(parsed.error().excerpt().empty());
    }
  }
}

TEST(ConfigIo, ScalarEnvelopeRequiresCanonicalNumericText) {
  // REQ: CONFIG-010
  constexpr std::string_view prefix = R"({"$holoscan.config.scalar/v1":{"kind":"f64","value":")";
  const auto envelope = [&](std::string_view value) {
    return std::string(prefix) + std::string(value) + R"("}})";
  };

  for (const auto format : {Format::kJson, Format::kYaml}) {
    EXPECT_TRUE(try_parse(envelope("1"), format));
    for (const std::string_view value : {"1.0", "1.00", "1e0", "-0"}) {
      const auto parsed = try_parse(envelope(value), format);
      ASSERT_FALSE(parsed) << value;
      EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kUnsupportedValue) << value;
      EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse) << value;
    }
  }

  auto negative_zero = config::ConfigValue::try_floating(-0.0);
  ASSERT_TRUE(negative_zero);
  auto document = config::Document::try_from_value(std::move(*negative_zero));
  ASSERT_TRUE(document);
  auto encoded = try_serialize(*document, Format::kJson);
  ASSERT_TRUE(encoded);
  EXPECT_EQ(*encoded, envelope("0"));
}

TEST(ConfigIo, JsonLimitsFailBeforeAcceptingTheNextNode) {
  // REQ: CONFIG-011
  ParseOptions options;
  options.max_nodes = 2;
  EXPECT_TRUE(try_parse("[1]", Format::kJson, options));
  EXPECT_EQ(try_parse("[1,2]", Format::kJson, options).error().code(),
            config::ConfigErrorCode::kLimitExceeded);
  options.max_nodes = 1'000'001;
  EXPECT_EQ(try_parse("null", Format::kJson, options).error().code(),
            config::ConfigErrorCode::kInvalidArgument);
}

TEST(ConfigIo, AdmittedYamlSubsetProducesTheSameSemanticIdentity) {
  // REQ: CONFIG-012, CONFIG-018
  auto yaml = try_parse("camera:\n  gain: 1.25\n  enabled: true\n", Format::kYaml);
  auto json = try_parse(R"({"camera":{"gain":1.25,"enabled":true}})", Format::kJson);
  ASSERT_TRUE(yaml);
  ASSERT_TRUE(json);
  EXPECT_EQ(yaml->digest().semantic.sha256, json->digest().semantic.sha256);

  auto anchored = try_parse("value: &anchor 1\nsame: *anchor\n", Format::kYaml);
  ASSERT_TRUE(anchored);
  EXPECT_EQ(anchored->try_at("/value")->try_as_signed_integer().value(), 1);
  EXPECT_EQ(anchored->try_at("/same")->try_as_signed_integer().value(), 1);

  auto quoted_marker = try_parse("value: \"anchor & alias * are text\"\n", Format::kYaml);
  ASSERT_TRUE(quoted_marker);
  EXPECT_EQ(quoted_marker->try_at("/value")->try_as_string().value(), "anchor & alias * are text");
}

TEST(ConfigIo, YamlModeRoundTripsCanonicalJsonScalarRoots) {
  // REQ: CONFIG-012, CONFIG-018
  for (const std::string_view source :
       {"null", "true", "false", R"("sensor")", "0", "-1", "1.25"}) {
    auto json = try_parse(source, Format::kJson);
    auto yaml = try_parse(source, Format::kYaml);
    ASSERT_TRUE(json) << source;
    ASSERT_TRUE(yaml) << source;
    EXPECT_EQ(yaml->digest().semantic.sha256, json->digest().semantic.sha256) << source;

    auto serialized = try_serialize(*json, Format::kYaml);
    ASSERT_TRUE(serialized) << source;
    auto reparsed = try_parse(*serialized, Format::kYaml);
    ASSERT_TRUE(reparsed) << *serialized;
    EXPECT_EQ(reparsed->digest().semantic.sha256, json->digest().semantic.sha256) << source;
  }
}

TEST(ConfigIo, YamlEmptyValuesMaterializeNullWithoutBorrowingFollowingTokens) {
  // REQ: CONFIG-013, CONFIG-018
  // Key setup: yaml-cpp reports the following token's mark for each absent value below.
  constexpr std::array<std::string_view, 7> sources{
      "a:\nb: 1\n",
      "outer:\n  a:\n  b: 1\n",
      "outer:\n  a:\nnext: 1\n",
      "a: # note\nb: 1\n",
      "a:\n\"b\": 1\n",
      "stages:\n  - source: a\n    destination:\n    width: 2\n",
      "s:\n  - 1\n  -\n  - 3\n"};
  for (const auto source : sources) {
    auto document = try_parse(source, Format::kYaml);
    ASSERT_TRUE(document) << source;
  }

  auto mapping = try_parse("a:\nb: 1\n", Format::kYaml);
  ASSERT_TRUE(mapping);
  EXPECT_EQ(mapping->try_at("/a")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(mapping->try_at("/b")->try_as_signed_integer().value(), 1);

  auto stages =
      try_parse("stages:\n  - source: a\n    destination:\n    width: 2\n", Format::kYaml);
  ASSERT_TRUE(stages);
  EXPECT_EQ(stages->try_at("/stages/0/destination")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(stages->try_at("/stages/0/width")->try_as_signed_integer().value(), 2);

  auto sequence = try_parse("s:\n  - 1\n  -\n  - 3\n", Format::kYaml);
  ASSERT_TRUE(sequence);
  EXPECT_EQ(sequence->try_at("/s/1")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(sequence->try_at("/s/2")->try_as_signed_integer().value(), 3);

  for (const auto rejected : {"a: Null\nb: 1\n", "a: NULL\nb: 1\n"}) {
    auto document = try_parse(rejected, Format::kYaml);
    ASSERT_FALSE(document) << rejected;
    EXPECT_EQ(document.error().code(), config::ConfigErrorCode::kUnsupportedValue);
  }
  auto lookalike = try_parse("a:\nNULLIFY: 1\n", Format::kYaml);
  ASSERT_TRUE(lookalike);
  EXPECT_EQ(lookalike->try_at("/a")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(lookalike->try_at("/NULLIFY")->try_as_signed_integer().value(), 1);

  constexpr std::string_view lossless_source = "empty:\nnext: 1\n";
  auto lossless =
      try_parse(lossless_source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(lossless);
  auto next = lossless->try_replace("/next", std::int32_t{2});
  ASSERT_TRUE(next);
  auto exact = try_serialize(*next, Format::kYaml);
  ASSERT_TRUE(exact);
  EXPECT_EQ(*exact, "empty:\nnext: 2\n");

  auto absent_edit = lossless->try_replace("/empty", config::ConfigValue::null());
  ASSERT_FALSE(absent_edit);
  EXPECT_EQ(absent_edit.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable);
}

TEST(ConfigIo, YamlNullRootsRetainScalarAndDocumentStreamPolicy) {
  // REQ: CONFIG-018
  for (const auto source : {"", "# comment only\n", "null", "~", "--- null\n", "&root null\n"}) {
    auto document = try_parse(source, Format::kYaml);
    ASSERT_TRUE(document) << source;
    EXPECT_EQ(document->root().kind(), config::ConfigValueKind::kNull) << source;
  }

  for (const auto source : {"Null",
                            "NULL",
                            "--- Null\n",
                            "&root NULL\n",
                            "---\na: 1\n--- null\n",
                            "--- null\n---\na: 1\n",
                            "null\n---\na: 1\n",
                            "value: abc 'def\n---\na: 2\n",
                            "value: abc \"def\n---\na: 2\n",
                            "value: abc 'def\r\n---\r\na: 2\r\n",
                            "value: abc 'def\n...\na: 2\n",
                            "null\n...\n...\n",
                            "value: foo:'bar\n---\na: 2\n",
                            "value: foo:\"bar\n---\na: 2\n",
                            "value: abc,'def\n---\na: 2\n",
                            "value: abc['def\n---\na: 2\n",
                            "value: abc{'def\n---\na: 2\n",
                            "value: ?'abc\n---\na: 2\n",
                            "values: [# , ' comment\n  1]\n---\nnext: 2\n",
                            "[1]# x: ' comment\n---\nnext: 2\n",
                            "\"x\"# : ' comment\n---\nnext: 2\n",
                            "value: foo:'bar\n%YAML 1.2\n---\na: 2\n"}) {
    auto document = try_parse(source, Format::kYaml);
    ASSERT_FALSE(document) << source;
    EXPECT_EQ(document.error().code(), config::ConfigErrorCode::kUnsupportedValue) << source;
  }

  auto quoted_directive = try_parse("\"line one\n%YAML literal\nline two\"\n", Format::kYaml);
  ASSERT_TRUE(quoted_directive);
  EXPECT_EQ(quoted_directive->root().try_as_string().value(), "line one %YAML literal line two");

  auto quoted_flow = try_parse("{\"key\":\"line one\n%YAML literal\nline two\"}\n", Format::kYaml);
  ASSERT_TRUE(quoted_flow);
  EXPECT_EQ(quoted_flow->try_at("/key")->try_as_string().value(),
            "line one %YAML literal line two");

  for (const auto source :
       {"values: [&a,'line\n%YAML literal\n']\n", "values: [&a,\"line\n%YAML literal\n\"]\n"}) {
    auto anchored_flow = try_parse(source, Format::kYaml);
    ASSERT_TRUE(anchored_flow) << source;
    const auto value = anchored_flow->try_at("/values/1")->try_as_string();
    ASSERT_TRUE(value) << source;
    EXPECT_NE(value->find("%YAML literal"), std::string::npos) << source;
  }
}

TEST(ConfigIo, YamlUtf8BomKeepsSourceMarksAligned) {
  // REQ: CONFIG-013, CONFIG-018
  constexpr std::string_view bom{"\xEF\xBB\xBF"};
  const std::string source = std::string(bom) + "name: sensor\ncount: 1\n";
  auto document =
      try_parse(source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);
  EXPECT_EQ(document->try_at("/name")->try_as_string().value(), "sensor");
  EXPECT_EQ(document->try_at("/count")->try_as_signed_integer().value(), 1);

  auto edited = document->try_replace("/name", std::string{"sensor_two"});
  ASSERT_TRUE(edited);
  auto exact = try_serialize(*edited, Format::kYaml);
  ASSERT_TRUE(exact);
  EXPECT_EQ(*exact, std::string(bom) + "name: sensor_two\ncount: 1\n");

  auto quoted_root = try_parse(std::string(bom) + "\"sensor\"\n", Format::kYaml);
  ASSERT_TRUE(quoted_root);
  EXPECT_EQ(quoted_root->root().try_as_string().value(), "sensor");

  auto exact_null = try_parse(
      std::string(bom) + "null\n", Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(exact_null);
  EXPECT_EQ(exact_null->root().kind(), config::ConfigValueKind::kNull);
  auto replayed = try_serialize(*exact_null, Format::kYaml);
  ASSERT_TRUE(replayed);
  EXPECT_EQ(*replayed, std::string(bom) + "null\n");

  auto rejected_null = try_parse(std::string(bom) + "Null\n", Format::kYaml);
  ASSERT_FALSE(rejected_null);
  EXPECT_EQ(rejected_null.error().code(), config::ConfigErrorCode::kUnsupportedValue);
  EXPECT_NE(rejected_null.error().message().find("line 1, column 1"), std::string_view::npos);
}

TEST(ConfigIo, YamlAnchoredScalarsPreserveContentAcrossLineBreaks) {
  // REQ: CONFIG-018
  constexpr std::string_view source = "value: &value\n  sensor\ncopy: *value\n";
  auto document = try_parse(source, Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_EQ(document->try_at("/value")->try_as_string().value(), "sensor");
  EXPECT_EQ(document->try_at("/copy")->try_as_string().value(), "sensor");

  auto lossless =
      try_parse(source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(lossless);
  auto replayed = try_serialize(*lossless, Format::kYaml);
  ASSERT_TRUE(replayed);
  EXPECT_EQ(*replayed, source);

  for (const auto spelling : {"null", "~"}) {
    const auto anchored = std::string("value: &value\n  ") + spelling + "\ncopy: *value\n";
    auto parsed = try_parse(anchored, Format::kYaml);
    ASSERT_TRUE(parsed) << anchored;
    EXPECT_EQ(parsed->try_at("/value")->kind(), config::ConfigValueKind::kNull);
    EXPECT_EQ(parsed->try_at("/copy")->kind(), config::ConfigValueKind::kNull);
  }
  for (const auto spelling : {"Null", "NULL"}) {
    const auto anchored = std::string("value: &value\n  ") + spelling + "\nnext: 1\n";
    auto parsed = try_parse(anchored, Format::kYaml);
    ASSERT_FALSE(parsed) << anchored;
    EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kUnsupportedValue);
    EXPECT_NE(parsed.error().message().find("line 2, column 3"), std::string_view::npos);
  }

  auto anchored_empty = try_parse("empty: &empty\nempty_copy: *empty\nnext: 1\n", Format::kYaml);
  ASSERT_TRUE(anchored_empty);
  EXPECT_EQ(anchored_empty->try_at("/empty")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(anchored_empty->try_at("/empty_copy")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(anchored_empty->try_at("/next")->try_as_signed_integer().value(), 1);

  auto commented =
      try_parse("value: &value\n  # retained separation\n  sensor\ncopy: *value\n", Format::kYaml);
  ASSERT_TRUE(commented);
  EXPECT_EQ(commented->try_at("/value")->try_as_string().value(), "sensor");
  EXPECT_EQ(commented->try_at("/copy")->try_as_string().value(), "sensor");
}

TEST(ConfigIo, YamlLosslessPlainStringEditsStayInsideTheAdmittedLanguage) {
  // REQ: CONFIG-013, CONFIG-018
  constexpr std::array<std::string_view, 8> rejected{
      "yes", "on", "no", "2026-07-29", "a ", " a", "1.2.3", "--verbose"};
  for (const auto replacement : rejected) {
    auto document = try_parse(
        "value: sensor\n", Format::kYaml, {.preservation = Preservation::kRequireLossless});
    ASSERT_TRUE(document) << replacement;
    auto edited = document->try_replace("/value", std::string{replacement});
    ASSERT_FALSE(edited) << replacement;
    EXPECT_EQ(edited.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable)
        << replacement;

    auto unchanged = try_serialize(*document, Format::kYaml);
    ASSERT_TRUE(unchanged) << replacement;
    EXPECT_EQ(*unchanged, "value: sensor\n") << replacement;
  }

  auto document =
      try_parse("value: sensor\n", Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);
  auto edited = document->try_replace("/value", std::string{"sensor_two"});
  ASSERT_TRUE(edited);
  auto emitted = try_serialize(*edited, Format::kYaml);
  ASSERT_TRUE(emitted);
  auto reparsed = try_parse(*emitted, Format::kYaml);
  ASSERT_TRUE(reparsed) << *emitted;
  EXPECT_EQ(reparsed->try_at("/value")->try_as_string().value(), "sensor_two");

  auto flow = try_parse(
      "values: [sensor]\n", Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(flow);
  for (const auto replacement : {"a,b", "a]b", "a{b"}) {
    auto unsafe = flow->try_replace("/values/0", std::string{replacement});
    ASSERT_FALSE(unsafe) << replacement;
    EXPECT_EQ(unsafe.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable)
        << replacement;
  }
  auto unchanged = try_serialize(*flow, Format::kYaml);
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(*unchanged, "values: [sensor]\n");
}

TEST(ConfigIo, YamlDirectivesCannotBeHiddenByCommentText) {
  // REQ: CONFIG-018
  for (const auto source : {"%YAML 1.2\n---\na: 1\n",
                            "# first\n# second\n%YAML 1.2\n---\na: 1\n",
                            "# first\n# don 't\n%YAML 1.2\n---\na: 1\n",
                            "# don 't\r%YAML 1.2\r---\ra: 1\r"}) {
    auto document = try_parse(source, Format::kYaml);
    ASSERT_FALSE(document) << source;
    EXPECT_EQ(document.error().code(), config::ConfigErrorCode::kUnsupportedValue) << source;
    EXPECT_EQ(document.error().operation_stage(), config::ConfigOperationStage::kParse) << source;
    EXPECT_TRUE(document.error().path().empty()) << source;
  }

  auto commented_directive = try_parse("# %YAML 1.2 is comment text\na: 1\n", Format::kYaml);
  ASSERT_TRUE(commented_directive);
  EXPECT_EQ(commented_directive->try_at("/a")->try_as_signed_integer().value(), 1);

  auto plain_content = try_parse("value: line one\n  %YAML literal\n", Format::kYaml);
  ASSERT_FALSE(plain_content);
  EXPECT_EQ(plain_content.error().path(), "/value");
  EXPECT_NE(plain_content.error().message().find("folded YAML plain scalars are unsupported"),
            std::string_view::npos);

  auto quoted_content =
      try_parse("items:\n  - one\n  - \"line\n    %YAML literal\n    two\"\n", Format::kYaml);
  ASSERT_TRUE(quoted_content);
  EXPECT_EQ(quoted_content->try_at("/items/1")->try_as_string().value(), "line %YAML literal two");

  auto cr_directive = try_parse("# don 't\r%YAML 1.2\r---\ra: 1\r", Format::kYaml);
  ASSERT_FALSE(cr_directive);
  EXPECT_NE(cr_directive.error().message().find("line 2, column 1"), std::string_view::npos);
}

TEST(ConfigIo, YamlQuotedScalarsAndKeysPreserveDecodedValuesAndComments) {
  // REQ: CONFIG-012, CONFIG-018
  // REQ: CONFIG-013
  constexpr std::string_view source = "value: \"sensor \\\"primary\\\" #1\" # retained comment\n";
  auto document =
      try_parse(source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);
  EXPECT_EQ(document->try_at("/value")->try_as_string().value(), "sensor \"primary\" #1");

  auto edited = document->try_replace("/value", std::string{"sensor #2"});
  ASSERT_TRUE(edited);
  auto exact = try_serialize(*edited, Format::kYaml);
  ASSERT_TRUE(exact);
  EXPECT_EQ(*exact, "value: \"sensor #2\" # retained comment\n");

  auto quoted_styles = try_parse(
      "'a/b~c': 'sensor #1'\n\"double key\": \"double value\"\n\"<<\": ordinary\n", Format::kYaml);
  ASSERT_TRUE(quoted_styles);
  EXPECT_EQ(quoted_styles->try_at("/a~1b~0c")->try_as_string().value(), "sensor #1");
  EXPECT_EQ(quoted_styles->try_at("/double key")->try_as_string().value(), "double value");
  EXPECT_EQ(quoted_styles->try_at("/<<")->try_as_string().value(), "ordinary");

  constexpr std::string_view retained_styles =
      "value: 'it''s #1' # retained single-quoted comment\n"
      "control: \"sensor #1\" # retained double-quoted comment\n";
  auto retained =
      try_parse(retained_styles, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(retained);
  EXPECT_EQ(retained->try_at("/value")->try_as_string().value(), "it's #1");
  EXPECT_EQ(retained->try_at("/control")->try_as_string().value(), "sensor #1");

  auto single_quoted = retained->try_replace("/value", std::string{"it's #2"});
  ASSERT_TRUE(single_quoted);
  auto both_quoted = single_quoted->try_replace("/control", std::string{"sensor #2"});
  ASSERT_TRUE(both_quoted);
  auto retained_exact = try_serialize(*both_quoted, Format::kYaml);
  ASSERT_TRUE(retained_exact);
  EXPECT_EQ(*retained_exact,
            "value: 'it''s #2' # retained single-quoted comment\n"
            "control: \"sensor #2\" # retained double-quoted comment\n");

  auto plain_successor = retained->try_replace("/value", std::string{"sensor two"});
  ASSERT_TRUE(plain_successor);
  auto plain_successor_exact = try_serialize(*plain_successor, Format::kYaml);
  ASSERT_TRUE(plain_successor_exact);
  EXPECT_EQ(*plain_successor_exact,
            "value: 'sensor two' # retained single-quoted comment\n"
            "control: \"sensor #1\" # retained double-quoted comment\n");
}

TEST(ConfigIo, YamlLosslessScalarEnvelopesPreserveKindsAndOutsideBytes) {
  // REQ: CONFIG-010, CONFIG-013, CONFIG-018
  constexpr std::string_view source =
      R"yaml({ "before" : "unchanged", "signed" : { "$holoscan.config.scalar/v1" : { "kind" : "i64", "value" : "-9" } }, "unsigned" : { "$holoscan.config.scalar/v1" : { "kind" : "u64", "value" : "18446744073709551615" } }, "floating" : { "$holoscan.config.scalar/v1" : { "kind" : "f64", "value" : "1.5" } }, "after" : "unchanged" }
)yaml";
  auto document =
      try_parse(source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);
  auto signed_value = document->try_at("/signed");
  auto unsigned_value = document->try_at("/unsigned");
  auto floating_value = document->try_at("/floating");
  ASSERT_TRUE(signed_value);
  ASSERT_TRUE(unsigned_value);
  ASSERT_TRUE(floating_value);
  EXPECT_EQ(signed_value->kind(), config::ConfigValueKind::kSignedInteger);
  EXPECT_EQ(unsigned_value->kind(), config::ConfigValueKind::kUnsignedInteger);
  EXPECT_EQ(floating_value->kind(), config::ConfigValueKind::kFloating);

  auto signed_edit = document->try_replace("/signed", std::int64_t{12});
  ASSERT_TRUE(signed_edit);
  auto unsigned_edit = signed_edit->try_replace("/unsigned", std::uint64_t{7});
  ASSERT_TRUE(unsigned_edit);
  auto floating_edit = unsigned_edit->try_replace("/floating", 2.0);
  ASSERT_TRUE(floating_edit);
  auto exact = try_serialize(*floating_edit, Format::kYaml);
  ASSERT_TRUE(exact);
  EXPECT_EQ(
      *exact,
      R"yaml({ "before" : "unchanged", "signed" : {"$holoscan.config.scalar/v1":{"kind":"i64","value":"12"}}, "unsigned" : {"$holoscan.config.scalar/v1":{"kind":"u64","value":"7"}}, "floating" : {"$holoscan.config.scalar/v1":{"kind":"f64","value":"2"}}, "after" : "unchanged" }
)yaml");

  auto reparsed = try_parse(*exact, Format::kYaml);
  ASSERT_TRUE(reparsed);
  auto reparsed_signed = reparsed->try_at("/signed");
  auto reparsed_unsigned = reparsed->try_at("/unsigned");
  auto reparsed_floating = reparsed->try_at("/floating");
  ASSERT_TRUE(reparsed_signed);
  ASSERT_TRUE(reparsed_unsigned);
  ASSERT_TRUE(reparsed_floating);
  ASSERT_EQ(reparsed_signed->kind(), config::ConfigValueKind::kSignedInteger);
  EXPECT_EQ(reparsed_signed->try_as_signed_integer().value(), 12);
  ASSERT_EQ(reparsed_unsigned->kind(), config::ConfigValueKind::kUnsignedInteger);
  EXPECT_EQ(reparsed_unsigned->try_as_unsigned_integer().value(), 7U);
  ASSERT_EQ(reparsed_floating->kind(), config::ConfigValueKind::kFloating);
  EXPECT_DOUBLE_EQ(reparsed_floating->try_as_floating().value(), 2.0);
}

TEST(ConfigIo, YamlFlowEnvelopeSpanIgnoresCommentDelimitersAtTokenBoundaries) {
  // REQ: CONFIG-013, CONFIG-018
  constexpr std::string_view source =
      R"yaml(value: {# } is comment text, not the end of the flow mapping
"$holoscan.config.scalar/v1" :# } is still comment text
{"kind": "i64", "value": "7"}
}
next: 1
)yaml";
  auto document =
      try_parse(source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);

  auto edited = document->try_replace("/value", std::int64_t{8});
  ASSERT_TRUE(edited);
  auto exact = try_serialize(*edited, Format::kYaml);
  ASSERT_TRUE(exact);
  EXPECT_EQ(*exact,
            "value: {\"$holoscan.config.scalar/v1\":{\"kind\":\"i64\",\"value\":\"8\"}}\n"
            "next: 1\n");

  auto reparsed = try_parse(*exact, Format::kYaml);
  ASSERT_TRUE(reparsed) << *exact;
  EXPECT_EQ(reparsed->try_at("/value")->try_as_signed_integer().value(), 8);
  EXPECT_EQ(reparsed->try_at("/next")->try_as_signed_integer().value(), 1);

  constexpr std::string_view cr_source =
      "value: {\r# } is comment text\r"
      "\"$holoscan.config.scalar/v1\": {\"kind\": \"i64\", \"value\": \"7\"}\r}\r"
      "next: 1\r";
  auto cr_document =
      try_parse(cr_source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(cr_document);
  auto cr_edited = cr_document->try_replace("/value", std::int64_t{8});
  ASSERT_TRUE(cr_edited);
  auto cr_exact = try_serialize(*cr_edited, Format::kYaml);
  ASSERT_TRUE(cr_exact);
  EXPECT_EQ(*cr_exact,
            "value: {\"$holoscan.config.scalar/v1\":{\"kind\":\"i64\",\"value\":\"8\"}}\r"
            "next: 1\r");
}

TEST(ConfigIo, YamlAliasesMaterializeIndependentOwningValues) {
  // REQ: CONFIG-018
  constexpr std::string_view source = R"yaml(defaults: &defaults
  width: 1920
  height: 1080
camera: *defaults
scalar: &scalar 7
scalar_copy: *scalar
steps: &steps [prepare, execute]
first_steps: *steps
second_steps: *steps
nested: &nested
  camera: *defaults
nested_copy: *nested
empty: &empty
empty_copy: *empty
)yaml";

  auto document = try_parse(source, Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_EQ(document->try_at("/defaults/width")->try_as_signed_integer().value(), 1920);
  EXPECT_EQ(document->try_at("/camera/height")->try_as_signed_integer().value(), 1080);
  EXPECT_EQ(document->try_at("/scalar_copy")->try_as_signed_integer().value(), 7);
  EXPECT_EQ(document->try_at("/first_steps/0")->try_as_string().value(), "prepare");
  EXPECT_EQ(document->try_at("/second_steps/1")->try_as_string().value(), "execute");
  EXPECT_EQ(document->try_at("/nested_copy/camera/width")->try_as_signed_integer().value(), 1920);
  EXPECT_EQ(document->try_at("/empty")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(document->try_at("/empty_copy")->kind(), config::ConfigValueKind::kNull);

  auto edited = document->try_replace("/camera/width", std::int32_t{1280});
  ASSERT_TRUE(edited);
  EXPECT_EQ(edited->try_at("/camera/width")->try_as_signed_integer().value(), 1280);
  EXPECT_EQ(edited->try_at("/defaults/width")->try_as_signed_integer().value(), 1920);
  EXPECT_EQ(document->try_at("/camera/width")->try_as_signed_integer().value(), 1920);

  auto comma_scalar = try_parse("label: &label alpha,beta\nuse: [*label]\n", Format::kYaml);
  ASSERT_TRUE(comma_scalar);
  EXPECT_EQ(comma_scalar->try_at("/label")->try_as_string().value(), "alpha,beta");
  EXPECT_EQ(comma_scalar->try_at("/use/0")->try_as_string().value(), "alpha,beta");
}

TEST(ConfigIo, YamlApplicationFixturesAdmitAliasesAndSequencesOfMappings) {
  // REQ: CONFIG-018
  constexpr std::string_view stages = R"yaml(stages:
  - source: stream_alpha
    destination: result_alpha
    width: 2
    samples: 6
  - source: stream_beta
    destination: result_beta
    width: 2
    samples: 6
  - source: stream_gamma
    destination: result_gamma
    width: 2
    samples: 6
  - source: stream_delta
    destination: result_delta
    width: 8
    samples: 6
  - source: stream_epsilon
    destination: result_epsilon
    width: 8
    samples: 6
  - source: stream_zeta
    destination: result_zeta
    width: 8
    samples: 6
  - source: stream_eta
    destination: result_eta
    width: 8
)yaml";

  auto focused = try_parse(stages, Format::kYaml);
  ASSERT_TRUE(focused);
  auto focused_items = focused->try_at("/stages");
  ASSERT_TRUE(focused_items);
  EXPECT_EQ(focused_items->kind(), config::ConfigValueKind::kArray);
  EXPECT_EQ(focused_items->size(), 7U);
  EXPECT_EQ(focused->try_at("/stages/0/source")->try_as_string().value(), "stream_alpha");
  EXPECT_EQ(focused->try_at("/stages/3/width")->try_as_signed_integer().value(), 8);
  EXPECT_EQ(focused->try_at("/stages/6/destination")->try_as_string().value(), "result_eta");

  auto application = try_parse(kSyntheticApplicationConfig, Format::kYaml);
  ASSERT_TRUE(application);
  auto anchor_order = application->try_at("/_anchors/ordered_channels");
  auto source_order = application->try_at("/source/channels");
  auto processor_order = application->try_at("/processor/channels");
  auto startup_order = application->try_at("/startup/channels");
  ASSERT_TRUE(anchor_order);
  ASSERT_TRUE(source_order);
  ASSERT_TRUE(processor_order);
  ASSERT_TRUE(startup_order);
  EXPECT_EQ(anchor_order->kind(), config::ConfigValueKind::kArray);
  EXPECT_EQ(anchor_order->size(), 5U);
  EXPECT_EQ(source_order->size(), 5U);
  EXPECT_EQ(processor_order->size(), 5U);
  EXPECT_EQ(startup_order->size(), 5U);
  EXPECT_EQ(application->try_at("/source/channels/0")->try_as_string().value(), "channel_alpha");
  EXPECT_EQ(application->try_at("/startup/channels/4")->try_as_string().value(), "channel_epsilon");
  EXPECT_EQ(application->try_at("/processor/stages")->size(), 7U);
  EXPECT_EQ(application->try_at("/processor/stages/0/note")->kind(),
            config::ConfigValueKind::kNull);
  EXPECT_EQ(application->try_at("/processor/stages/5/width")->try_as_signed_integer().value(), 8);
}

TEST(ConfigIo, YamlCyclesAndMergeKeysFailClosed) {
  // REQ: CONFIG-018
  struct CycleCase {
    std::string_view source;
    std::string_view path;
  };
  constexpr std::array cycles{
      CycleCase{"self: &self [*self]\n", "/self/0"},
      CycleCase{"outer: &outer\n  inner: &inner\n    outer: *outer\ncopy: *inner\n",
                "/outer/inner/outer"},
  };
  for (const auto& item : cycles) {
    auto cycle = try_parse(item.source, Format::kYaml);
    ASSERT_FALSE(cycle) << item.source;
    EXPECT_EQ(cycle.error().code(), config::ConfigErrorCode::kUnsupportedValue) << item.source;
    EXPECT_EQ(cycle.error().token(), "CONFIG_UNSUPPORTED_VALUE") << item.source;
    EXPECT_EQ(cycle.error().operation_stage(), config::ConfigOperationStage::kParse) << item.source;
    EXPECT_EQ(cycle.error().path(), item.path) << item.source;
    // yaml-cpp exposes the anchor-definition mark, not the alias-use mark. The converter must
    // retain the semantic use path without fabricating a source excerpt for that unavailable mark.
    EXPECT_TRUE(cycle.error().excerpt().empty()) << item.source;
  }

  constexpr std::string_view merge_source = R"yaml(defaults: &defaults
  width: 1920
camera:
  <<: *defaults
  height: 1080
)yaml";
  auto merge = try_parse(merge_source, Format::kYaml);
  ASSERT_FALSE(merge);
  EXPECT_EQ(merge.error().code(), config::ConfigErrorCode::kUnsupportedValue);
  EXPECT_EQ(merge.error().token(), "CONFIG_UNSUPPORTED_VALUE");
  EXPECT_EQ(merge.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_EQ(merge.error().path(), "/camera/<<");
  EXPECT_FALSE(merge.error().excerpt().empty());

  auto quoted_merge = try_parse("camera:\n  \"<<\": literal\n", Format::kYaml);
  ASSERT_TRUE(quoted_merge);
  EXPECT_EQ(quoted_merge->try_at("/camera/<<")->try_as_string().value(), "literal");

  auto undefined = try_parse("value: *missing\n", Format::kYaml);
  ASSERT_FALSE(undefined);
  EXPECT_EQ(undefined.error().code(), config::ConfigErrorCode::kParseError);
  EXPECT_EQ(undefined.error().token(), "CONFIG_PARSE_ERROR");
  EXPECT_EQ(undefined.error().operation_stage(), config::ConfigOperationStage::kParse);

  auto repeated = try_parse("value: &value [1, 2]\nfirst: *value\nsecond: *value\n", Format::kYaml);
  ASSERT_TRUE(repeated);
  EXPECT_EQ(repeated->try_at("/first/0")->try_as_signed_integer().value(), 1);
  EXPECT_EQ(repeated->try_at("/second/1")->try_as_signed_integer().value(), 2);
}

TEST(ConfigIo, YamlAliasExpansionHonorsEveryParseLimit) {
  // REQ: CONFIG-018
  constexpr std::string_view source = "s: &s [aa]\nu: [*s, *s]\n";

  ParseOptions options;
  options.max_source_bytes = source.size();
  EXPECT_TRUE(try_parse(source, Format::kYaml, options));
  options.max_source_bytes = source.size() - 1U;
  auto source_limit = try_parse(source, Format::kYaml, options);
  ASSERT_FALSE(source_limit);
  EXPECT_EQ(source_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);

  options = {};
  options.max_depth = 4;
  EXPECT_TRUE(try_parse(source, Format::kYaml, options));
  options.max_depth = 3;
  auto depth_limit = try_parse(source, Format::kYaml, options);
  ASSERT_FALSE(depth_limit);
  EXPECT_EQ(depth_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(depth_limit.error().path(), "/u/0/0");

  // The materialized tree contains eight values: the root, the anchored sequence and scalar,
  // the uses sequence, and two independently expanded sequence/scalar pairs.
  options = {};
  options.max_nodes = 8;
  EXPECT_TRUE(try_parse(source, Format::kYaml, options));
  options.max_nodes = 7;
  auto node_limit = try_parse(source, Format::kYaml, options);
  ASSERT_FALSE(node_limit);
  EXPECT_EQ(node_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(node_limit.error().path(), "/u/1/0");

  options = {};
  options.max_string_bytes = 2;
  EXPECT_TRUE(try_parse(source, Format::kYaml, options));
  options.max_string_bytes = 1;
  auto string_limit = try_parse(source, Format::kYaml, options);
  ASSERT_FALSE(string_limit);
  EXPECT_EQ(string_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(string_limit.error().path(), "/s/0");
  EXPECT_FALSE(string_limit.error().excerpt().empty());

  options = {};
  options.max_container_entries = 2;
  EXPECT_TRUE(try_parse(source, Format::kYaml, options));
  options.max_container_entries = 1;
  auto entry_limit = try_parse(source, Format::kYaml, options);
  ASSERT_FALSE(entry_limit);
  EXPECT_EQ(entry_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_TRUE(entry_limit.error().path().empty());
  EXPECT_FALSE(entry_limit.error().excerpt().empty());

  constexpr std::string_view key_source = "m: &m {long: x}\nu: [*m, *m]\n";
  options = {};
  options.max_string_bytes = 4;
  EXPECT_TRUE(try_parse(key_source, Format::kYaml, options));
  options.max_string_bytes = 3;
  auto key_limit = try_parse(key_source, Format::kYaml, options);
  ASSERT_FALSE(key_limit);
  EXPECT_EQ(key_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(key_limit.error().path(), "/m");
  EXPECT_FALSE(key_limit.error().excerpt().empty());

  constexpr std::string_view exponential = "a: &a [x]\nb: &b [*a, *a]\nc: &c [*b, *b]\nroot: *c\n";
  options = {};
  options.max_nodes = 30;
  EXPECT_TRUE(try_parse(exponential, Format::kYaml, options));
  options.max_nodes = 29;
  auto expansion_limit = try_parse(exponential, Format::kYaml, options);
  ASSERT_FALSE(expansion_limit);
  EXPECT_EQ(expansion_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(expansion_limit.error().token(), "CONFIG_LIMIT_EXCEEDED");
  EXPECT_EQ(expansion_limit.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_EQ(expansion_limit.error().path(), "/root/1/1/0");

  auto materialized = try_parse(source, Format::kYaml);
  ASSERT_TRUE(materialized);
  SerializeOptions output_options;
  output_options.output_limits.max_nodes = 8;
  EXPECT_TRUE(try_serialize(*materialized, Format::kYaml, output_options));
  output_options.output_limits.max_nodes = 7;
  auto output_node_limit = try_serialize(*materialized, Format::kYaml, output_options);
  ASSERT_FALSE(output_node_limit);
  EXPECT_EQ(output_node_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(output_node_limit.error().operation_stage(), config::ConfigOperationStage::kSerialize);

  auto canonical = try_serialize(*materialized, Format::kYaml);
  ASSERT_TRUE(canonical);
  output_options = {};
  output_options.output_limits.max_output_bytes = canonical->size();
  EXPECT_TRUE(try_serialize(*materialized, Format::kYaml, output_options));
  output_options.output_limits.max_output_bytes = canonical->size() - 1U;
  auto output_byte_limit = try_serialize(*materialized, Format::kYaml, output_options);
  ASSERT_FALSE(output_byte_limit);
  EXPECT_EQ(output_byte_limit.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(output_byte_limit.error().operation_stage(), config::ConfigOperationStage::kSerialize);
}

TEST(ConfigIo, YamlNodeConversionPreservesScalarAndDiagnosticPolicies) {
  // REQ: CONFIG-018
  constexpr std::string_view admitted = R"yaml(signed: -1
unsigned: 18446744073709551615
floating: -0.0
boolean: true
null_value: null
plain: sensor
dot_prefixed: .config
plus_prefixed: +sensor
signed_dot_prefixed: +.config
+.config: signed_dot_key
single_quoted: '1'
double_quoted: "true"
"a/b~c": 'decoded key'
block:
  - 1
  - name: sensor
flow: [1, {"name": 'sensor'}]
)yaml";
  auto document = try_parse(admitted, Format::kYaml);
  ASSERT_TRUE(document);
  EXPECT_EQ(document->try_at("/signed")->kind(), config::ConfigValueKind::kSignedInteger);
  EXPECT_EQ(document->try_at("/unsigned")->kind(), config::ConfigValueKind::kUnsignedInteger);
  EXPECT_EQ(document->try_at("/unsigned")->try_as_unsigned_integer().value(),
            std::numeric_limits<std::uint64_t>::max());
  EXPECT_EQ(document->try_at("/floating")->kind(), config::ConfigValueKind::kFloating);
  EXPECT_EQ(document->try_at("/floating")->try_as_floating().value(), 0.0);
  EXPECT_FALSE(std::signbit(document->try_at("/floating")->try_as_floating().value()));
  EXPECT_EQ(document->try_at("/boolean")->kind(), config::ConfigValueKind::kBool);
  EXPECT_EQ(document->try_at("/null_value")->kind(), config::ConfigValueKind::kNull);
  EXPECT_EQ(document->try_at("/plain")->try_as_string().value(), "sensor");
  EXPECT_EQ(document->try_at("/dot_prefixed")->try_as_string().value(), ".config");
  EXPECT_EQ(document->try_at("/plus_prefixed")->try_as_string().value(), "+sensor");
  EXPECT_EQ(document->try_at("/signed_dot_prefixed")->try_as_string().value(), "+.config");
  EXPECT_EQ(document->try_at("/+.config")->try_as_string().value(), "signed_dot_key");
  EXPECT_EQ(document->try_at("/single_quoted")->try_as_string().value(), "1");
  EXPECT_EQ(document->try_at("/double_quoted")->try_as_string().value(), "true");
  EXPECT_EQ(document->try_at("/a~1b~0c")->try_as_string().value(), "decoded key");
  EXPECT_EQ(document->try_at("/block/1/name")->try_as_string().value(), "sensor");
  EXPECT_EQ(document->try_at("/flow/1/name")->try_as_string().value(), "sensor");

  constexpr std::string_view duplicate = R"yaml(history_terms:
  - dim: 14
    dim: 15
)yaml";
  auto duplicate_result = try_parse(duplicate, Format::kYaml);
  ASSERT_FALSE(duplicate_result);
  EXPECT_EQ(duplicate_result.error().code(), config::ConfigErrorCode::kDuplicateKey);
  EXPECT_EQ(duplicate_result.error().token(), "CONFIG_DUPLICATE_KEY");
  EXPECT_EQ(duplicate_result.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_EQ(duplicate_result.error().path(), "/history_terms/0/dim");
  EXPECT_NE(error_excerpt(duplicate_result.error()).find("dim: 15"), std::string::npos);
  EXPECT_NE(duplicate_result.error().message().find("line"), std::string_view::npos);
  EXPECT_NE(duplicate_result.error().message().find("column"), std::string_view::npos);

  auto decoded_duplicate = try_parse("\"dim\": 14\ndim: 15\n", Format::kYaml);
  ASSERT_FALSE(decoded_duplicate);
  EXPECT_EQ(decoded_duplicate.error().code(), config::ConfigErrorCode::kDuplicateKey);
  EXPECT_EQ(decoded_duplicate.error().path(), "/dim");

  constexpr std::array<std::string_view, 5> numeric_policy_keys{
      "01", "+1", "0x10", "1_000", "1:20"};
  for (const std::string_view key : numeric_policy_keys) {
    const auto source = std::string(key) + ": value\n";
    auto parsed = try_parse(source, Format::kYaml);
    ASSERT_FALSE(parsed) << source;
    EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kUnsupportedValue) << source;
    EXPECT_EQ(parsed.error().token(), "CONFIG_UNSUPPORTED_VALUE") << source;
    EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse) << source;
    EXPECT_TRUE(parsed.error().path().empty()) << source;
    EXPECT_FALSE(parsed.error().excerpt().empty()) << source;
  }

  constexpr std::string_view quoted_policy_keys = R"yaml("01": leading_zero
'+1': leading_plus
"0x10": base_prefix
'1_000': underscore
"1:20": sexagesimal
)yaml";
  auto quoted_keys = try_parse(quoted_policy_keys, Format::kYaml);
  ASSERT_TRUE(quoted_keys);
  EXPECT_EQ(quoted_keys->try_at("/01")->try_as_string().value(), "leading_zero");
  EXPECT_EQ(quoted_keys->try_at("/+1")->try_as_string().value(), "leading_plus");
  EXPECT_EQ(quoted_keys->try_at("/0x10")->try_as_string().value(), "base_prefix");
  EXPECT_EQ(quoted_keys->try_at("/1_000")->try_as_string().value(), "underscore");
  EXPECT_EQ(quoted_keys->try_at("/1:20")->try_as_string().value(), "sexagesimal");

  struct RejectedPolicyCase {
    std::string_view source;
    config::ConfigErrorCode code;
    std::string_view token;
    std::string_view path;
  };
  constexpr std::array rejected{
      RejectedPolicyCase{"? [a, b]\n: value\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         ""},
      RejectedPolicyCase{"value: !!str 1\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: !custom sensor\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: 2026-07-29\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: !!binary SGVsbG8=\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"---\nvalue: 1\n---\nvalue: 2\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         ""},
      RejectedPolicyCase{"%YAML 1.2\n---\nvalue: 1\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         ""},
      RejectedPolicyCase{"value: +1\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: 01\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: 1_000\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: 0x10\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: 1:20\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: .inf\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: NaN\n",
                         config::ConfigErrorCode::kUnsupportedValue,
                         "CONFIG_UNSUPPORTED_VALUE",
                         "/value"},
      RejectedPolicyCase{"value: 18446744073709551616\n",
                         config::ConfigErrorCode::kOutOfRange,
                         "CONFIG_OUT_OF_RANGE",
                         "/value"},
  };
  for (const auto& item : rejected) {
    auto parsed = try_parse(item.source, Format::kYaml);
    ASSERT_FALSE(parsed) << item.source;
    EXPECT_EQ(parsed.error().code(), item.code) << item.source;
    EXPECT_EQ(parsed.error().token(), item.token) << item.source;
    EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse)
        << item.source;
    EXPECT_EQ(parsed.error().path(), item.path) << item.source;
    EXPECT_FALSE(parsed.error().excerpt().empty()) << item.source;
  }

  constexpr std::string_view offset_directive = "# preface\n%YAML 1.2\n---\nvalue: 1\n";
  auto directive = try_parse(offset_directive, Format::kYaml);
  ASSERT_FALSE(directive);
  EXPECT_EQ(directive.error().code(), config::ConfigErrorCode::kUnsupportedValue);
  EXPECT_EQ(directive.error().token(), "CONFIG_UNSUPPORTED_VALUE");
  EXPECT_EQ(directive.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_TRUE(directive.error().path().empty());
  EXPECT_EQ(error_excerpt(directive.error()), "%YAML 1.2\n---\nvalue: 1\n");
  EXPECT_NE(directive.error().message().find("line 2"), std::string_view::npos);
  EXPECT_NE(directive.error().message().find("column 1"), std::string_view::npos);

  constexpr std::string_view bom_directive =
      "\xEF\xBB\xBF"
      "%YAML 1.2\n---\nvalue: 1\n";
  auto bom = try_parse(bom_directive, Format::kYaml);
  ASSERT_FALSE(bom);
  EXPECT_EQ(bom.error().code(), config::ConfigErrorCode::kUnsupportedValue);
  EXPECT_EQ(bom.error().token(), "CONFIG_UNSUPPORTED_VALUE");
  EXPECT_EQ(bom.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_TRUE(bom.error().path().empty());
  EXPECT_EQ(error_excerpt(bom.error()), "%YAML 1.2\n---\nvalue: 1\n");

  constexpr std::array nonspecific_tags{
      std::pair<std::string_view, std::string_view>{"value: ! sensor\n", "column 8"},
      std::pair<std::string_view, std::string_view>{"value: &v ! sensor\n", "column 11"},
  };
  for (const auto& [source, expected_column] : nonspecific_tags) {
    auto tagged = try_parse(source, Format::kYaml);
    ASSERT_FALSE(tagged) << source;
    EXPECT_EQ(tagged.error().code(), config::ConfigErrorCode::kUnsupportedValue) << source;
    EXPECT_EQ(tagged.error().token(), "CONFIG_UNSUPPORTED_VALUE") << source;
    EXPECT_EQ(tagged.error().operation_stage(), config::ConfigOperationStage::kParse) << source;
    EXPECT_EQ(tagged.error().path(), "/value") << source;
    EXPECT_EQ(error_excerpt(tagged.error()), "! sensor\n") << source;
    EXPECT_NE(tagged.error().message().find(expected_column), std::string_view::npos) << source;
  }

  constexpr std::string_view anchored_tag = "value: &v !custom sensor\n";
  auto explicit_tag = try_parse(anchored_tag, Format::kYaml);
  ASSERT_FALSE(explicit_tag);
  EXPECT_EQ(explicit_tag.error().code(), config::ConfigErrorCode::kUnsupportedValue);
  EXPECT_EQ(explicit_tag.error().token(), "CONFIG_UNSUPPORTED_VALUE");
  EXPECT_EQ(explicit_tag.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_EQ(explicit_tag.error().path(), "/value");
  EXPECT_EQ(error_excerpt(explicit_tag.error()), "!custom sensor\n");
  EXPECT_NE(explicit_tag.error().message().find("line 1"), std::string_view::npos);
  EXPECT_NE(explicit_tag.error().message().find("column 11"), std::string_view::npos);

  constexpr std::string_view malformed_directive = "%YAML 1.2\n---\nvalue: [1,\n";
  auto malformed_policy = try_parse(malformed_directive, Format::kYaml);
  ASSERT_FALSE(malformed_policy);
  EXPECT_EQ(malformed_policy.error().code(), config::ConfigErrorCode::kParseError);
  EXPECT_EQ(malformed_policy.error().token(), "CONFIG_PARSE_ERROR");
  EXPECT_EQ(malformed_policy.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_TRUE(malformed_policy.error().path().empty());
  EXPECT_NE(error_excerpt(malformed_policy.error()).find("value: [1,"), std::string::npos);

  for (const std::string_view malformed : {"value: [1,\n", "value: *missing\n"}) {
    auto parsed = try_parse(malformed, Format::kYaml);
    ASSERT_FALSE(parsed) << malformed;
    EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kParseError) << malformed;
    EXPECT_EQ(parsed.error().token(), "CONFIG_PARSE_ERROR") << malformed;
    EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse) << malformed;
    EXPECT_FALSE(parsed.error().excerpt().empty()) << malformed;
  }

  std::string invalid_utf8{"value: \""};
  invalid_utf8.push_back(static_cast<char>(0xC3));
  invalid_utf8 += "(\"\n";
  auto invalid = try_parse(invalid_utf8, Format::kYaml);
  ASSERT_FALSE(invalid);
  EXPECT_EQ(invalid.error().code(), config::ConfigErrorCode::kInvalidUtf8);
  EXPECT_EQ(invalid.error().token(), "CONFIG_INVALID_UTF8");
  EXPECT_EQ(invalid.error().operation_stage(), config::ConfigOperationStage::kParse);
  EXPECT_EQ(invalid.error().path(), "/value");
  EXPECT_FALSE(invalid.error().excerpt().empty());
  EXPECT_LE(invalid.error().excerpt().size(), 256U);

  const auto expect_invalid_source_utf8 = [](const std::string& source) {
    auto parsed = try_parse(source, Format::kYaml);
    ASSERT_FALSE(parsed);
    EXPECT_EQ(parsed.error().code(), config::ConfigErrorCode::kInvalidUtf8);
    EXPECT_EQ(parsed.error().token(), "CONFIG_INVALID_UTF8");
    EXPECT_EQ(parsed.error().operation_stage(), config::ConfigOperationStage::kParse);
    EXPECT_TRUE(parsed.error().path().empty());
    const auto excerpt = error_excerpt(parsed.error());
    ASSERT_FALSE(excerpt.empty());
    EXPECT_EQ(static_cast<unsigned char>(excerpt.front()), 0xC3U);
    EXPECT_EQ(excerpt.size(), 256U);
  };

  std::string invalid_comment{"value: 1 # "};
  invalid_comment.push_back(static_cast<char>(0xC3));
  invalid_comment.append(300U, 'x');
  invalid_comment.push_back('\n');
  expect_invalid_source_utf8(invalid_comment);

  std::string invalid_anchor{"value: &anchor"};
  invalid_anchor.push_back(static_cast<char>(0xC3));
  invalid_anchor.append(300U, 'x');
  invalid_anchor += " sensor\n";
  expect_invalid_source_utf8(invalid_anchor);

  const auto diagnostic_path =
      std::filesystem::temp_directory_path() /
      ("holoscan-config-yaml-diagnostic-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".yaml");
  {
    std::ofstream output(diagnostic_path, std::ios::binary);
    output << duplicate;
  }
  auto loaded_duplicate = try_load(diagnostic_path, Format::kYaml);
  EXPECT_FALSE(loaded_duplicate);
  if (!loaded_duplicate) {
    EXPECT_EQ(loaded_duplicate.error().source(), diagnostic_path.string());
    EXPECT_EQ(loaded_duplicate.error().path(), "/history_terms/0/dim");
    EXPECT_NE(error_excerpt(loaded_duplicate.error()).find("dim: 15"), std::string::npos);
  }
  std::filesystem::remove(diagnostic_path);
}

TEST(ConfigIo, YamlAliasIdentitySeparatesSemanticAndRawDigests) {
  // REQ: CONFIG-018
  constexpr std::string_view alias_source = R"yaml(template: &item
  width: 1920
  labels: [left, right]
first: *item
second: *item
)yaml";
  constexpr std::string_view expanded_source = R"yaml(template:
  width: 1920
  labels: [left, right]
first:
  width: 1920
  labels: [left, right]
second:
  width: 1920
  labels: [left, right]
)yaml";
  constexpr std::string_view expanded_json =
      R"({"template":{"width":1920,"labels":["left","right"]},"first":{"width":1920,"labels":["left","right"]},"second":{"width":1920,"labels":["left","right"]}})";

  auto aliases = try_parse(alias_source, Format::kYaml);
  auto expanded = try_parse(expanded_source, Format::kYaml);
  auto json = try_parse(expanded_json, Format::kJson);
  ASSERT_TRUE(aliases);
  ASSERT_TRUE(expanded);
  ASSERT_TRUE(json);
  EXPECT_EQ(aliases->digest().semantic, expanded->digest().semantic);
  EXPECT_EQ(aliases->digest().semantic, json->digest().semantic);
  ASSERT_TRUE(aliases->digest().source);
  ASSERT_TRUE(expanded->digest().source);
  EXPECT_NE(aliases->digest().source, expanded->digest().source);

  auto alias_bytes = config::try_encode_semantic_value_v1(aliases->root());
  auto expanded_bytes = config::try_encode_semantic_value_v1(expanded->root());
  ASSERT_TRUE(alias_bytes);
  ASSERT_TRUE(expanded_bytes);
  EXPECT_EQ(*alias_bytes, *expanded_bytes);

  auto canonical_json = try_serialize(*aliases, Format::kJson);
  auto canonical_yaml = try_serialize(*aliases, Format::kYaml);
  ASSERT_TRUE(canonical_json);
  ASSERT_TRUE(canonical_yaml);
  EXPECT_EQ(*canonical_json, *canonical_yaml);
  EXPECT_EQ(canonical_yaml->find('&'), std::string::npos);
  EXPECT_EQ(canonical_yaml->find('*'), std::string::npos);
  EXPECT_EQ(canonical_yaml->find("<<"), std::string::npos);
  auto reparsed = try_parse(*canonical_yaml, Format::kYaml);
  ASSERT_TRUE(reparsed);
  EXPECT_EQ(reparsed->digest().semantic, aliases->digest().semantic);
}

TEST(ConfigIo, YamlAliasLosslessReplayRejectsEdits) {
  // REQ: CONFIG-018
  constexpr std::string_view source = R"yaml(defaults: &defaults
  width: 1920
  height: 1080
camera: *defaults # exact alias spelling and comment must survive unchanged replay
)yaml";

  auto lossless =
      try_parse(source, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(lossless);
  auto replay = try_serialize(*lossless, Format::kYaml);
  ASSERT_TRUE(replay);
  EXPECT_EQ(*replay, source);
  auto cross_format = try_serialize(*lossless, Format::kJson);
  ASSERT_FALSE(cross_format);
  EXPECT_EQ(cross_format.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable);

  auto edit = lossless->try_replace("/camera/width", std::int32_t{1280});
  ASSERT_FALSE(edit);
  EXPECT_EQ(edit.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable);
  EXPECT_EQ(edit.error().token(), "CONFIG_LOSSLESS_PRESERVATION_UNAVAILABLE");

  auto semantic = try_parse(source, Format::kYaml);
  ASSERT_TRUE(semantic);
  auto semantic_edit = semantic->try_replace("/camera/width", std::int32_t{1280});
  ASSERT_TRUE(semantic_edit);
  EXPECT_EQ(semantic_edit->try_at("/camera/width")->try_as_signed_integer().value(), 1280);
  EXPECT_EQ(semantic_edit->try_at("/defaults/width")->try_as_signed_integer().value(), 1920);
  auto canonical = try_serialize(*semantic_edit, Format::kYaml);
  ASSERT_TRUE(canonical);
  EXPECT_EQ(canonical->find('&'), std::string::npos);
  EXPECT_EQ(canonical->find('*'), std::string::npos);

  constexpr std::string_view marker_text =
      "value: \"literal *alias and &anchor text\"\n"
      "plain_marker: abc-*def\n"
      "url: http:*foo\n"
      "count: 1 # comment containing *alias and &anchor markers\n";
  auto ordinary_lossless =
      try_parse(marker_text, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(ordinary_lossless);
  EXPECT_EQ(ordinary_lossless->try_at("/plain_marker")->try_as_string().value(), "abc-*def");
  EXPECT_EQ(ordinary_lossless->try_at("/url")->try_as_string().value(), "http:*foo");
  auto ordinary_edit = ordinary_lossless->try_replace("/count", std::int32_t{2});
  ASSERT_TRUE(ordinary_edit);
  auto ordinary_replay = try_serialize(*ordinary_edit, Format::kYaml);
  ASSERT_TRUE(ordinary_replay);
  EXPECT_EQ(*ordinary_replay,
            "value: \"literal *alias and &anchor text\"\n"
            "plain_marker: abc-*def\n"
            "url: http:*foo\n"
            "count: 2 # comment containing *alias and &anchor markers\n");
}

TEST(ConfigIo, InvalidClosedEnumsFailValidationInsteadOfSelectingAnotherBackend) {
  // REQ: CONFIG-010, CONFIG-014
  // Intentionally manufacture an invalid closed-enum value to verify the public fail-closed
  // boundary instead of relying on an unavailable enumerator.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_EQ(try_parse("null", static_cast<Format>(99)).error().code(),
            config::ConfigErrorCode::kInvalidArgument);
  ParseOptions parse_options;
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  parse_options.preservation = static_cast<Preservation>(99);
  EXPECT_EQ(try_parse("null", Format::kJson, parse_options).error().code(),
            config::ConfigErrorCode::kInvalidArgument);
}

TEST(ConfigIo, LosslessUnchangedRoundTripIsExplicitAndCrossFormatFailsClosed) {
  // REQ: CONFIG-013
  constexpr std::string_view source = "{ \"value\" : 1 }\n";
  auto document =
      try_parse(source, Format::kJson, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);
  auto exact = try_serialize(*document, Format::kJson);
  ASSERT_TRUE(exact);
  EXPECT_EQ(*exact, source);
  auto cross_format = try_serialize(*document, Format::kYaml);
  EXPECT_FALSE(cross_format);
  EXPECT_EQ(cross_format.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable);
}

TEST(ConfigIo, LosslessSameKindScalarEditsPreserveEveryOtherSourceByte) {
  // REQ: CONFIG-013
  // REQ: CONFIG-019
  // Key setup: parse with kRequireLossless to enable byte-level source tracking for both
  //            JSON and YAML; edit multiple scalar values of matching kinds.
  constexpr std::string_view json = "{ \"first\" : 1, \"nested\" : [ true, \"old\" ] }\n";
  auto document = try_parse(json, Format::kJson, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(document);
  auto first = document->try_replace("/first", std::int32_t{27});
  ASSERT_TRUE(first);
  auto flag = first->try_replace("/nested/0", false);
  ASSERT_TRUE(flag);
  auto text = flag->try_replace("/nested/1", std::string{"new"});
  ASSERT_TRUE(text);
  auto exact_json = try_serialize(*text, Format::kJson);
  ASSERT_TRUE(exact_json);
  // Key assertion: only the replaced positions differ; all surrounding source bytes are intact.
  EXPECT_EQ(*exact_json, "{ \"first\" : 27, \"nested\" : [ false, \"new\" ] }\n");
  auto rewritten_digest = config::detail::raw_digest(*exact_json);
  ASSERT_TRUE(rewritten_digest);
  EXPECT_EQ(text->digest().source, *rewritten_digest);
  EXPECT_NE(text->digest().source, document->digest().source);
  EXPECT_EQ(text->digest().base_source, document->digest().source);

  constexpr std::string_view yaml = "first: 1\nnested:\n  - true\n  - old\n";
  auto yaml_document =
      try_parse(yaml, Format::kYaml, {.preservation = Preservation::kRequireLossless});
  ASSERT_TRUE(yaml_document);
  auto yaml_first = yaml_document->try_replace("/first", std::int32_t{9});
  ASSERT_TRUE(yaml_first);
  auto yaml_text = yaml_first->try_replace("/nested/1", std::string{"new"});
  ASSERT_TRUE(yaml_text);
  auto exact_yaml = try_serialize(*yaml_text, Format::kYaml);
  ASSERT_TRUE(exact_yaml);
  EXPECT_EQ(*exact_yaml, "first: 9\nnested:\n  - true\n  - new\n");
  auto rewritten_yaml_digest = config::detail::raw_digest(*exact_yaml);
  ASSERT_TRUE(rewritten_yaml_digest);
  EXPECT_EQ(yaml_text->digest().source, *rewritten_yaml_digest);
  EXPECT_EQ(yaml_text->digest().base_source, yaml_document->digest().source);

  // Key assertion: changing value kind or using an unsupported YAML style fails with
  //               kLosslessPreservationUnavailable.
  auto changed_kind = yaml_document->try_replace("/first", std::string{"one"});
  EXPECT_FALSE(changed_kind);
  EXPECT_EQ(changed_kind.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable);
  auto unsupported_style = yaml_document->try_replace("/nested/1", std::string{"new: value"});
  EXPECT_FALSE(unsupported_style);
  EXPECT_EQ(unsupported_style.error().code(),
            config::ConfigErrorCode::kLosslessPreservationUnavailable);

  auto replacement = config::ConfigValue::try_signed_integer(2);
  ASSERT_TRUE(replacement);
  auto first_path = config::Path::try_parse("/first");
  ASSERT_TRUE(first_path);
  auto extra = config::ConfigValue::try_boolean(true);
  ASSERT_TRUE(extra);
  auto extra_path = config::Path::try_parse("/extra");
  ASSERT_TRUE(extra_path);
  const std::array mixed_edits{config::Edit::replace(*first_path, *replacement),
                               config::Edit::add(*extra_path, *extra)};
  auto structural = yaml_document->try_patched(mixed_edits);
  EXPECT_FALSE(structural);
  EXPECT_EQ(structural.error().code(), config::ConfigErrorCode::kLosslessPreservationUnavailable);
}

TEST(ConfigIo, EditedDocumentsDistinguishCurrentAndBaseSourceIdentity) {
  // REQ: CONFIG-004
  // REQ: CONFIG-019
  constexpr std::string_view source = R"({"value":1})";
  auto document = try_parse(source, Format::kJson);
  ASSERT_TRUE(document);
  const auto original_digest = document->digest();
  ASSERT_TRUE(original_digest.source.has_value());
  EXPECT_FALSE(original_digest.base_source.has_value());

  auto edited = document->try_replace("/value", std::int32_t{2});
  ASSERT_TRUE(edited);
  EXPECT_FALSE(edited->digest().source.has_value());
  EXPECT_EQ(edited->digest().base_source, original_digest.source);

  auto edited_again = edited->try_replace("/value", std::int32_t{3});
  ASSERT_TRUE(edited_again);
  EXPECT_FALSE(edited_again->digest().source.has_value());
  EXPECT_EQ(edited_again->digest().base_source, original_digest.source);

  auto flag = config::ConfigValue::try_boolean(true);
  ASSERT_TRUE(flag);
  auto enabled_path = config::Path::try_parse("/enabled");
  ASSERT_TRUE(enabled_path);
  const std::array structural_edit{config::Edit::add(*enabled_path, *flag)};
  auto structural = document->try_patched(structural_edit);
  ASSERT_TRUE(structural);
  EXPECT_FALSE(structural->digest().source.has_value());
  EXPECT_EQ(structural->digest().base_source, original_digest.source);

  auto unchanged = document->try_patched({});
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(unchanged->digest().source, original_digest.source);
  EXPECT_EQ(unchanged->digest().base_source, original_digest.base_source);
}

TEST(ConfigIo, MissingInputReportsPathNotFound) {
  // REQ: CONFIG-004
  // Key setup: Select a unique path without creating the corresponding configuration file.
  const auto path =
      std::filesystem::temp_directory_path() /
      ("holoscan-config-missing-" +
       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");

  const auto loaded = try_load(path, Format::kJson);

  // Key assertion: File-open diagnostics distinguish an absent source from a generic I/O fault.
  ASSERT_FALSE(loaded);
  EXPECT_EQ(loaded.error().code(), config::ConfigErrorCode::kPathNotFound);
}

TEST(ConfigIo, GuardedAtomicSaveUsesRawCasAndReturnsNextDigest) {
  // REQ: CONFIG-014
  // REQ: CONFIG-019
  // Key setup: write an initial file, load it, replace one field, then save using the
  //            original source digest as the compare-and-swap guard token.
#if defined(_WIN32)
  GTEST_SKIP() << "Native Windows adapter is an explicit remaining qualification gate";
#else
  const auto root = std::filesystem::temp_directory_path() /
                    ("holoscan-config-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(root);
  const auto path = root / "config.json";
  {
    std::ofstream output(path, std::ios::binary);
    output << "{\"value\":1}";
  }
  auto loaded = try_load(path, Format::kJson);
  ASSERT_TRUE(loaded);
  auto updated = loaded->try_replace("/value", std::int32_t{2});
  ASSERT_TRUE(updated);
  ASSERT_FALSE(updated->digest().source.has_value());
  ASSERT_EQ(updated->digest().base_source, loaded->digest().source);
  auto saved = try_save(
      *updated,
      path,
      Format::kJson,
      {.expected_source_digest = updated->digest().base_source, .durability = Durability::kData});
  // Key assertion: first save succeeds and reports written_source_digest; second save
  //               with the now-stale original digest fails with kSourceChanged.
  ASSERT_TRUE(saved) << saved.error().message();
  EXPECT_EQ(saved->written_source_digest.byte_size, std::filesystem::file_size(path));
  auto stale = try_save(
      *updated,
      path,
      Format::kJson,
      {.expected_source_digest = updated->digest().base_source, .durability = Durability::kData});
  EXPECT_FALSE(stale);
  EXPECT_EQ(stale.error().code(), config::ConfigErrorCode::kSourceChanged);
  EXPECT_EQ(stale.error().operation_stage(), config::ConfigOperationStage::kAcquireGuard);
  EXPECT_EQ(stale.error().destination_state(), config::ConfigDestinationState::kUnchanged);
  std::filesystem::remove_all(root);
#endif
}

TEST(ConfigIo, GuardedAtomicSaveRejectsSymlinkedParentComponents) {
  // REQ: CONFIG-014
#if defined(_WIN32)
  GTEST_SKIP() << "Native Windows adapter is an explicit remaining qualification gate";
#else
  const auto root = std::filesystem::temp_directory_path() /
                    ("holoscan-config-symlink-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto real_parent = root / "real";
  const auto alias_parent = root / "alias";
  std::filesystem::create_directories(real_parent);
  std::filesystem::create_directory_symlink(real_parent, alias_parent);
  auto value = config::ConfigValue::try_signed_integer(1);
  ASSERT_TRUE(value);
  auto document = config::Document::try_from_value(std::move(*value));
  ASSERT_TRUE(document);
  auto saved = try_save(*document, alias_parent / "config.json", Format::kJson);
  EXPECT_FALSE(saved);
  EXPECT_EQ(saved.error().code(), config::ConfigErrorCode::kAtomicReplaceUnsupported);
  EXPECT_FALSE(std::filesystem::exists(real_parent / "config.json"));
  std::filesystem::remove_all(root);
#endif
}

TEST(ConfigIo, GuardDeadlineAndNoFollowDestinationFailBeforeWriting) {
  // REQ: CONFIG-014
  // Key setup: exercises two independent pre-write guard failures — (1) destination path
  //            is a symlink (kAtomicReplaceUnsupported), and (2) advisory lockfile is
  //            held with a zero-millisecond timeout (kGuardBusy).
#if defined(_WIN32)
  GTEST_SKIP() << "Native Windows adapter is an explicit remaining qualification gate";
#else
  const auto root = std::filesystem::temp_directory_path() /
                    ("holoscan-config-guard-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(root);
  const auto path = root / "config.json";
  const auto target = root / "target.json";
  {
    std::ofstream output(target, std::ios::binary);
    output << "{}";
  }
  std::filesystem::create_symlink(target.filename(), path);
  auto document = try_parse("{}", Format::kJson);
  ASSERT_TRUE(document);
  // Key assertion: symlink destination is rejected before any write attempt.
  auto symlink = try_save(*document, path, Format::kJson);
  ASSERT_FALSE(symlink);
  EXPECT_EQ(symlink.error().code(), config::ConfigErrorCode::kAtomicReplaceUnsupported);
  EXPECT_EQ(symlink.error().operation_stage(), config::ConfigOperationStage::kAcquireGuard);

  std::filesystem::remove(path);
  {
    std::ofstream output(path, std::ios::binary);
    output << "{}";
  }
  const auto lock_path = root / ".config.json.holoscan-config.lock";
  const int lock = ::open(lock_path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
  ASSERT_GE(lock, 0);
  ASSERT_EQ(::flock(lock, LOCK_EX | LOCK_NB), 0);
  SaveOptions options;
  options.guard_timeout = std::chrono::milliseconds{0};
  auto busy = try_save(*document, path, Format::kJson, options);
  ASSERT_FALSE(busy);
  // Key assertion: zero-timeout save returns kGuardBusy and leaves the destination unchanged.
  EXPECT_EQ(busy.error().code(), config::ConfigErrorCode::kGuardBusy);
  EXPECT_EQ(busy.error().operation_stage(), config::ConfigOperationStage::kAcquireGuard);
  EXPECT_EQ(busy.error().destination_state(), config::ConfigDestinationState::kUnchanged);
  ::close(lock);
  std::filesystem::remove_all(root);
#endif
}

TEST(ConfigIo, GuardFileSymlinkReportsAtomicReplaceUnsupported) {
  // REQ: CONFIG-014
#if defined(_WIN32)
  GTEST_SKIP() << "Native Windows adapter is an explicit remaining qualification gate";
#else
  const auto root = std::filesystem::temp_directory_path() /
                    ("holoscan-config-guard-symlink-test-" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(root);
  const auto path = root / "config.json";
  const auto target = root / "guard-target";
  std::ofstream{target} << "guard";
  std::filesystem::create_symlink(target.filename(), root / ".config.json.holoscan-config.lock");
  auto document = try_parse("{}", Format::kJson);
  ASSERT_TRUE(document);

  auto saved = try_save(*document, path, Format::kJson);

  ASSERT_FALSE(saved);
  EXPECT_EQ(saved.error().code(), config::ConfigErrorCode::kAtomicReplaceUnsupported);
  EXPECT_EQ(saved.error().operation_stage(), config::ConfigOperationStage::kAcquireGuard);
  EXPECT_EQ(saved.error().destination_state(), config::ConfigDestinationState::kUnchanged);
  EXPECT_FALSE(std::filesystem::exists(path));
  std::filesystem::remove_all(root);
#endif
}

}  // namespace
}  // namespace holoscan::config_io::test
