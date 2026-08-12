/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/config/document.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace holoscan::config::test {
namespace {

TEST(ConfigDocument, ValueAlgebraPreservesKindsAndRejectsInvalidAggregates) {
  // REQ: CONFIG-001
  EXPECT_EQ(ConfigValue::null().kind(), ConfigValueKind::kNull);
  ASSERT_TRUE(ConfigValue::boolean(true));
  EXPECT_EQ(ConfigValue::boolean(true)->kind(), ConfigValueKind::kBool);
  ASSERT_TRUE(ConfigValue::signed_integer(1));
  EXPECT_EQ(ConfigValue::signed_integer(1)->kind(), ConfigValueKind::kSignedInteger);
  ASSERT_TRUE(ConfigValue::unsigned_integer(1));
  EXPECT_EQ(ConfigValue::unsigned_integer(1)->kind(), ConfigValueKind::kUnsignedInteger);
  ASSERT_TRUE(ConfigValue::floating(-0.0));
  EXPECT_EQ(ConfigValue::floating(-0.0)->view().as_floating().value(), 0.0);
  EXPECT_FALSE(ConfigValue::floating(std::numeric_limits<double>::infinity()));
  EXPECT_EQ(ConfigValue::floating(std::numeric_limits<double>::infinity()).error().code(),
            ConfigErrorCode::kUnsupportedValue);
  const std::string invalid_utf8{"\xC0\x80", 2};
  EXPECT_EQ(ConfigValue::string(invalid_utf8).error().code(), ConfigErrorCode::kInvalidUtf8);

  auto one = ConfigValue::string("one");
  ASSERT_TRUE(one);
  auto flag = ConfigValue::boolean(false);
  ASSERT_TRUE(flag);
  const std::array duplicate{ConfigObjectEntry{"key", *one}, ConfigObjectEntry{"key", *flag}};
  EXPECT_EQ(ConfigValue::object(duplicate).error().code(), ConfigErrorCode::kDuplicateKey);
}

TEST(ConfigDocument, PathOwnsAndRoundTripsRfc6901Tokens) {
  // REQ: CONFIG-002
  std::string temporary{"/devices/camera~1front/gain~0mode"};
  auto path = Path::parse(temporary);
  temporary.clear();
  ASSERT_TRUE(path);
  ASSERT_EQ(path->token_count(), 3U);
  EXPECT_EQ(path->token(1), "camera/front");
  EXPECT_EQ(path->token(2), "gain~mode");
  EXPECT_EQ(path->to_string(), "/devices/camera~1front/gain~0mode");
  EXPECT_TRUE(Path::parse("")->is_root());
  EXPECT_EQ(Path::parse("not-root").error().code(), ConfigErrorCode::kPathSyntax);
  EXPECT_EQ(Path::parse("/~2").error().code(), ConfigErrorCode::kPathSyntax);
}

TEST(ConfigDocument, EditsAreImmutableTransactionalAndRejectOverlap) {
  // REQ: CONFIG-003
  // Key setup: two-field object document used for all immutability and overlap checks.
  auto value = ConfigValue::string("before");
  ASSERT_TRUE(value);
  auto enabled = ConfigValue::boolean(true);
  ASSERT_TRUE(enabled);
  std::array entries{ConfigObjectEntry{"name", *value}, ConfigObjectEntry{"enabled", *enabled}};
  auto root = ConfigValue::object(entries);
  ASSERT_TRUE(root);
  auto source_result = Document::from_value(*root);
  ASSERT_TRUE(source_result);
  const Document source = std::move(*source_result);
  auto name_path = Path::parse("/name");
  ASSERT_TRUE(name_path);
  auto after = ConfigValue::string("after");
  ASSERT_TRUE(after);
  const Edit replace = Edit::replace(*name_path, *after);
  auto updated = source.patched(std::span<const Edit>{&replace, 1});
  ASSERT_TRUE(updated);
  // Key assertion: source is unchanged after the patch; updated carries the new value.
  EXPECT_EQ(source.at("/name")->as_string().value(), "before");
  EXPECT_EQ(updated->at("/name")->as_string().value(), "after");
  EXPECT_NE(source.digest().semantic.sha256, updated->digest().semantic.sha256);

  auto child_path = Path::parse("/name/child");
  ASSERT_TRUE(child_path);
  std::array overlap{replace, Edit::remove(*child_path)};
  auto failed = source.patched(overlap);
  // Key assertion: overlapping edits (replace /name then remove /name/child) fail with
  //               kPathConflict and leave source unchanged.
  EXPECT_FALSE(failed);
  EXPECT_EQ(failed.error().code(), ConfigErrorCode::kPathConflict);
  EXPECT_EQ(source.at("/name")->as_string().value(), "before");
}

TEST(ConfigDocument, SemanticEncodingIsTypedAndObjectOrderIndependent) {
  // REQ: CONFIG-004
  // Key setup: two objects with the same key-value pairs in different insertion order.
  auto signed_one = ConfigValue::signed_integer(1);
  auto unsigned_one = ConfigValue::unsigned_integer(1);
  ASSERT_TRUE(signed_one);
  ASSERT_TRUE(unsigned_one);
  std::array first_entries{ConfigObjectEntry{"z", *signed_one},
                           ConfigObjectEntry{"a", *unsigned_one}};
  std::array second_entries{ConfigObjectEntry{"a", *unsigned_one},
                            ConfigObjectEntry{"z", *signed_one}};
  auto first = ConfigValue::object(first_entries);
  auto second = ConfigValue::object(second_entries);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto first_bytes = encode_semantic_value_v1(first->view());
  auto second_bytes = encode_semantic_value_v1(second->view());
  ASSERT_TRUE(first_bytes);
  ASSERT_TRUE(second_bytes);
  EXPECT_EQ(*first_bytes, *second_bytes);
  auto first_document = Document::from_value(*first);
  auto second_document = Document::from_value(*second);
  auto signed_document = Document::from_value(*signed_one);
  auto unsigned_document = Document::from_value(*unsigned_one);
  ASSERT_TRUE(first_document);
  ASSERT_TRUE(second_document);
  ASSERT_TRUE(signed_document);
  ASSERT_TRUE(unsigned_document);
  // Key assertion: objects with identical content but different key order share the same
  //               semantic digest; i64 vs u64 representation of "1" differs.
  EXPECT_EQ(first_document->digest().semantic.sha256, second_document->digest().semantic.sha256);
  EXPECT_NE(signed_document->digest().semantic.sha256, unsigned_document->digest().semantic.sha256);
}

TEST(ConfigDocument, BoundedErrorUsesStableTokensAndDeterministicTruncation) {
  // REQ: CONFIG-005
  const std::string oversize(ConfigError::kMaxMessageBytes + 10U, 'x');
  auto error = ConfigError::make(ConfigErrorCode::kGuardBusy, oversize);
  EXPECT_EQ(error.token(), "CONFIG_GUARD_BUSY");
  EXPECT_EQ(error.message().size(), ConfigError::kMaxMessageBytes);
  EXPECT_TRUE(error.message_truncated());
  EXPECT_EQ(error.operation_stage(), ConfigOperationStage::kNone);
  const ConfigError copy = error;
  EXPECT_EQ(copy.message(), error.message());
}

TEST(ConfigDocument, ErrorDomainOwnsExactTokensStagesAndBoundedCauseChain) {
  // REQ: CONFIG-005
  constexpr std::array<std::string_view, 22> expected_tokens{
      "CONFIG_INVALID_ARGUMENT",
      "CONFIG_PARSE_ERROR",
      "CONFIG_UNSUPPORTED_FORMAT",
      "CONFIG_UNSUPPORTED_VALUE",
      "CONFIG_INVALID_UTF8",
      "CONFIG_DUPLICATE_KEY",
      "CONFIG_PATH_SYNTAX",
      "CONFIG_PATH_NOT_FOUND",
      "CONFIG_PATH_CONFLICT",
      "CONFIG_TYPE_MISMATCH",
      "CONFIG_OUT_OF_RANGE",
      "CONFIG_LIMIT_EXCEEDED",
      "CONFIG_LOSSLESS_PRESERVATION_UNAVAILABLE",
      "CONFIG_SOURCE_CHANGED",
      "CONFIG_GUARD_BUSY",
      "CONFIG_CODEC_MISSING",
      "CONFIG_CODEC_FAILURE",
      "CONFIG_IO_ERROR",
      "CONFIG_PERMISSION_DENIED",
      "CONFIG_ATOMIC_REPLACE_UNSUPPORTED",
      "CONFIG_DIGEST_MISMATCH",
      "CONFIG_UNSUPPORTED_VERSION"};
  // Key assertion: every ConfigErrorCode maps to a stable SCREAMING_SNAKE_CASE token.
  for (std::size_t index = 0; index < expected_tokens.size(); ++index) {
    EXPECT_EQ(config_error_token(static_cast<ConfigErrorCode>(index)), expected_tokens[index]);
  }

  constexpr std::array expected_stages{ConfigOperationStage::kNone,
                                       ConfigOperationStage::kParse,
                                       ConfigOperationStage::kValidate,
                                       ConfigOperationStage::kSerialize,
                                       ConfigOperationStage::kWriteTemporary,
                                       ConfigOperationStage::kAcquireGuard,
                                       ConfigOperationStage::kReplaceDestination,
                                       ConfigOperationStage::kFlushDestinationMetadata};
  for (std::size_t index = 0; index < expected_stages.size(); ++index) {
    EXPECT_EQ(static_cast<std::uint8_t>(expected_stages[index]), index);
  }

  constexpr std::array expected_destination_states{ConfigDestinationState::kNotApplicable,
                                                   ConfigDestinationState::kUnchanged,
                                                   ConfigDestinationState::kReplaced};
  for (std::size_t index = 0; index < expected_destination_states.size(); ++index) {
    EXPECT_EQ(static_cast<std::uint8_t>(expected_destination_states[index]), index);
  }

  // Key setup: nest more errors than kMaxNestedCauseDepth to verify truncation behavior.
  ConfigError chain = ConfigError::make(ConfigErrorCode::kParseError, "leaf", "/leaf");
  for (std::size_t index = 0; index < ConfigError::kMaxNestedCauseDepth + 2U; ++index) {
    chain = ConfigError::nest(
        ConfigError::make(ConfigErrorCode::kCodecFailure, "wrapper", "/wrapper"), chain);
  }
  // Key assertion: cause chain is capped at kMaxNestedCauseDepth and overflow is flagged.
  EXPECT_EQ(chain.cause_count(), ConfigError::kMaxNestedCauseDepth);
  EXPECT_TRUE(chain.cause_overflow());
  ASSERT_NE(chain.cause(ConfigError::kMaxNestedCauseDepth - 1U), nullptr);
  EXPECT_TRUE(chain.cause(ConfigError::kMaxNestedCauseDepth - 1U)->cause_overflow());
  EXPECT_EQ(chain.cause(ConfigError::kMaxNestedCauseDepth), nullptr);
  EXPECT_EQ(ConfigError::make(ConfigErrorCode::kSourceChanged,
                              {},
                              {},
                              {},
                              {},
                              ConfigOperationStage::kAcquireGuard,
                              ConfigDestinationState::kUnchanged)
                .operation_stage(),
            ConfigOperationStage::kAcquireGuard);
}

}  // namespace
}  // namespace holoscan::config::test
