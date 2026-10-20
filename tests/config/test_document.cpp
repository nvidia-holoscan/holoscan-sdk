/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/config/document.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../common/allocation_counter.hpp"

namespace holoscan::config::test {
namespace {

TEST(ConfigDocument, ThrowingValueAndViewOperationsKeepOptionalAbsenceNormal) {
  // REQ: CONFIG-020
  const ConfigValue flag = ConfigValue::boolean(false);
  EXPECT_FALSE(flag.view().as_bool());

  auto wrong_type = flag.view().try_as_string();
  ASSERT_FALSE(wrong_type);
  EXPECT_EQ(wrong_type.error().code(), ConfigErrorCode::kTypeMismatch);
  EXPECT_THROW(static_cast<void>(flag.view().as_string()), ConfigException);

  const ConfigValue object = ConfigValue::object(std::span<const ConfigObjectEntry>{});
  EXPECT_FALSE(object.view().find_member("missing").has_value());
  auto missing = object.view().try_find_member("missing");
  ASSERT_TRUE(missing);
  EXPECT_FALSE(missing->has_value());

  try {
    static_cast<void>(ConfigValue::floating(std::numeric_limits<double>::infinity()));
    FAIL() << "expected non-finite value rejection";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kUnsupportedValue);
  }
}

TEST(ConfigDocument, ThrowingSemanticHelpersUseTheStructuredBridge) {
  // REQ: CONFIG-020
  const ConfigValue baseline = ConfigValue::boolean(false);
  const ConfigValue patch = ConfigValue::boolean(true);
  EXPECT_TRUE(merge_config_values(baseline, patch).view().as_bool());
  EXPECT_FALSE(encode_semantic_value_v1(patch.view()).empty());

  auto failed = try_encode_semantic_value_v1(ConfigView{});
  ASSERT_FALSE(failed);
  EXPECT_EQ(failed.error().code(), ConfigErrorCode::kInvalidArgument);
  try {
    static_cast<void>(encode_semantic_value_v1(ConfigView{}));
    FAIL() << "expected invalid view rejection";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kInvalidArgument);
  }
}

TEST(ConfigDocument, ThrowingDocumentOperationsPreserveOptionalLookup) {
  // REQ: CONFIG-020
  const std::array entries{ConfigObjectEntry{"enabled", ConfigValue::boolean(true)}};
  const Document document = Document::from_value(ConfigValue::object(entries));

  EXPECT_TRUE(document.at("/enabled").as_bool());
  auto missing = document.try_at("/missing");
  ASSERT_FALSE(missing);
  EXPECT_EQ(missing.error().code(), ConfigErrorCode::kPathNotFound);
  try {
    static_cast<void>(document.at("/missing"));
    FAIL() << "expected missing path rejection";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kPathNotFound);
  }

  EXPECT_FALSE(document.find("/missing").has_value());
  const Path scalar_child = Path::parse("/enabled/value");
  auto invalid_owned_traversal = document.try_find(scalar_child);
  ASSERT_FALSE(invalid_owned_traversal);
  EXPECT_EQ(invalid_owned_traversal.error().code(), ConfigErrorCode::kTypeMismatch);
  auto invalid_string_traversal = document.try_find("/enabled/value");
  ASSERT_FALSE(invalid_string_traversal);
  EXPECT_EQ(invalid_string_traversal.error().code(), ConfigErrorCode::kTypeMismatch);
  EXPECT_THROW(static_cast<void>(document.find(scalar_child)), ConfigException);
  EXPECT_THROW(static_cast<void>(document.find("/enabled/value")), ConfigException);
  auto malformed = document.try_find("not-rooted");
  ASSERT_FALSE(malformed);
  EXPECT_EQ(malformed.error().code(), ConfigErrorCode::kPathSyntax);

  const Path long_missing = Path::parse("/a-missing-path-longer-than-small-string-storage");
  const holoscan::test::ScopedAllocationFailure fail_next_allocation{0U};
  auto allocation_failure = document.try_at(long_missing);
  ASSERT_FALSE(allocation_failure);
  EXPECT_EQ(allocation_failure.error().code(), ConfigErrorCode::kLimitExceeded);

  const Document updated =
      document.patched(Edit::replace(Path::parse("/enabled"), ConfigValue::boolean(false)));
  EXPECT_FALSE(updated.at("/enabled").as_bool());
}

TEST(ConfigDocument, ValueAlgebraPreservesKindsAndRejectsInvalidAggregates) {
  // REQ: CONFIG-001
  EXPECT_EQ(ConfigValue::null().kind(), ConfigValueKind::kNull);
  ASSERT_TRUE(ConfigValue::try_boolean(true));
  EXPECT_EQ(ConfigValue::try_boolean(true)->kind(), ConfigValueKind::kBool);
  ASSERT_TRUE(ConfigValue::try_signed_integer(1));
  EXPECT_EQ(ConfigValue::try_signed_integer(1)->kind(), ConfigValueKind::kSignedInteger);
  ASSERT_TRUE(ConfigValue::try_unsigned_integer(1));
  EXPECT_EQ(ConfigValue::try_unsigned_integer(1)->kind(), ConfigValueKind::kUnsignedInteger);
  ASSERT_TRUE(ConfigValue::try_floating(-0.0));
  EXPECT_EQ(ConfigValue::try_floating(-0.0)->view().try_as_floating().value(), 0.0);
  EXPECT_FALSE(ConfigValue::try_floating(std::numeric_limits<double>::infinity()));
  EXPECT_EQ(ConfigValue::try_floating(std::numeric_limits<double>::infinity()).error().code(),
            ConfigErrorCode::kUnsupportedValue);
  const std::string invalid_utf8{"\xC0\x80", 2};
  EXPECT_EQ(ConfigValue::try_string(invalid_utf8).error().code(), ConfigErrorCode::kInvalidUtf8);

  auto one = ConfigValue::try_string("one");
  ASSERT_TRUE(one);
  auto flag = ConfigValue::try_boolean(false);
  ASSERT_TRUE(flag);
  const std::array duplicate{ConfigObjectEntry{"key", *one}, ConfigObjectEntry{"key", *flag}};
  EXPECT_EQ(ConfigValue::try_object(duplicate).error().code(), ConfigErrorCode::kDuplicateKey);
}

TEST(ConfigDocument, PathOwnsAndRoundTripsRfc6901Tokens) {
  // REQ: CONFIG-002
  std::string temporary{"/devices/camera~1front/gain~0mode"};
  auto path = Path::try_parse(temporary);
  temporary.clear();
  ASSERT_TRUE(path);
  ASSERT_EQ(path->token_count(), 3U);
  EXPECT_EQ(path->token(1), "camera/front");
  EXPECT_EQ(path->token(2), "gain~mode");
  EXPECT_EQ(path->to_string(), "/devices/camera~1front/gain~0mode");
  EXPECT_TRUE(Path::parse("").is_root());

  auto malformed = Path::try_parse("not-root");
  ASSERT_FALSE(malformed);
  EXPECT_EQ(malformed.error().code(), ConfigErrorCode::kPathSyntax);
}

TEST(ConfigDocument, PathParseThrowsConfigException) {
  // REQ: CONFIG-002
  // REQ: CONFIG-020
  try {
    static_cast<void>(Path::parse("/~2"));
    FAIL() << "Path::parse accepted malformed syntax";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kPathSyntax);
    EXPECT_EQ(exception.what(), format_error(exception.error()));
  }
}

TEST(ConfigDocument, ConfigExceptionPreservesStructuredError) {
  // REQ: CONFIG-005
  // REQ: CONFIG-020
  const std::string message(ConfigError::kMaxMessageBytes + 1U, 'm');
  const std::array excerpt{std::byte{'b'}, std::byte{'a'}, std::byte{'d'}};
  const ConfigError cause = ConfigError::make(
      ConfigErrorCode::kTypeMismatch, "expected object", "/model", "model.yaml", excerpt);
  const ConfigError error =
      ConfigError::nest(ConfigError::make(ConfigErrorCode::kIoError,
                                          message,
                                          "/gr00t_n1_7/inference",
                                          "application.yaml",
                                          excerpt,
                                          ConfigOperationStage::kFlushDestinationMetadata,
                                          ConfigDestinationState::kReplaced),
                        cause);

  try {
    static_cast<void>(detail::value_or_throw(ConfigResult<int>{tl::make_unexpected(error)}));
    FAIL() << "expected ConfigException";
  } catch (const ConfigException& exception) {
    const ConfigError& retained = exception.error();
    EXPECT_EQ(retained.code(), error.code());
    EXPECT_EQ(retained.message(), error.message());
    EXPECT_EQ(retained.path(), error.path());
    EXPECT_EQ(retained.source(), error.source());
    EXPECT_TRUE(std::ranges::equal(retained.excerpt(), error.excerpt()));
    EXPECT_TRUE(retained.message_truncated());
    EXPECT_FALSE(retained.cause_overflow());
    EXPECT_EQ(retained.operation_stage(), ConfigOperationStage::kFlushDestinationMetadata);
    EXPECT_EQ(retained.destination_state(), ConfigDestinationState::kReplaced);
    ASSERT_EQ(retained.cause_count(), 1U);
    ASSERT_NE(retained.cause(0), nullptr);
    EXPECT_EQ(retained.cause(0)->code(), cause.code());
    EXPECT_EQ(retained.cause(0)->message(), cause.message());
    EXPECT_EQ(retained.cause(0)->path(), cause.path());
    EXPECT_EQ(retained.cause(0)->source(), cause.source());
    EXPECT_EQ(retained.cause(0)->operation_stage(), ConfigOperationStage::kNone);
    EXPECT_EQ(retained.cause(0)->destination_state(), ConfigDestinationState::kNotApplicable);
    EXPECT_EQ(exception.what(), format_error(exception.error()));
  }
}

TEST(ConfigDocument, EditFactoriesAcceptStringPathsAndViews) {
  // REQ: CONFIG-003
  // REQ: CONFIG-020
  const ConfigValue scalar = ConfigValue::string("payload");
  const std::array entries{ConfigObjectEntry{"field", scalar}};
  const ConfigValue object = ConfigValue::object(entries);

  const std::array edits{Edit::add("/added", scalar.view()),
                         Edit::replace("/replaced", scalar),
                         Edit::remove("/removed"),
                         Edit::merge_object("/merged", object.view())};
  EXPECT_EQ(edits[0].kind(), EditKind::kAdd);
  EXPECT_EQ(edits[0].path().to_string(), "/added");
  EXPECT_EQ(edits[0].value().view().as_string(), "payload");
  EXPECT_EQ(edits[1].kind(), EditKind::kReplace);
  EXPECT_EQ(edits[1].path().to_string(), "/replaced");
  EXPECT_EQ(edits[2].kind(), EditKind::kRemove);
  EXPECT_EQ(edits[2].path().to_string(), "/removed");
  EXPECT_EQ(edits[3].kind(), EditKind::kMergeObject);
  EXPECT_EQ(edits[3].path().to_string(), "/merged");
  EXPECT_EQ(edits[3].value().kind(), ConfigValueKind::kObject);

  const auto expect_edit =
      [](const ConfigResult<Edit>& result, EditKind kind, std::string_view path) {
        ASSERT_TRUE(result);
        EXPECT_EQ(result->kind(), kind);
        EXPECT_EQ(result->path().to_string(), path);
      };
  expect_edit(Edit::try_add("/added", scalar.view()), EditKind::kAdd, "/added");
  expect_edit(Edit::try_replace("/replaced", scalar), EditKind::kReplace, "/replaced");
  expect_edit(Edit::try_remove("/removed"), EditKind::kRemove, "/removed");
  expect_edit(Edit::try_merge_object("/merged", object.view()), EditKind::kMergeObject, "/merged");

  auto add_failure = Edit::try_add("invalid", scalar);
  auto replace_failure = Edit::try_replace("invalid", scalar.view());
  auto remove_failure = Edit::try_remove("invalid");
  auto merge_failure = Edit::try_merge_object("invalid", object);
  ASSERT_FALSE(add_failure);
  ASSERT_FALSE(replace_failure);
  ASSERT_FALSE(remove_failure);
  ASSERT_FALSE(merge_failure);
  EXPECT_EQ(add_failure.error().code(), ConfigErrorCode::kPathSyntax);
  EXPECT_EQ(replace_failure.error().code(), add_failure.error().code());
  EXPECT_EQ(remove_failure.error().code(), add_failure.error().code());
  EXPECT_EQ(merge_failure.error().code(), add_failure.error().code());
  try {
    static_cast<void>(Edit::remove("invalid"));
    FAIL() << "invalid edit path did not throw";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kPathSyntax);
  }
}

TEST(ConfigDocument, EditsAreImmutableTransactionalAndRejectOverlap) {
  // REQ: CONFIG-019
  // REQ: CONFIG-003
  // Key setup: two-field object document used for all immutability and overlap checks.
  auto value = ConfigValue::try_string("before");
  ASSERT_TRUE(value);
  auto enabled = ConfigValue::try_boolean(true);
  ASSERT_TRUE(enabled);
  std::array entries{ConfigObjectEntry{"name", *value}, ConfigObjectEntry{"enabled", *enabled}};
  auto root = ConfigValue::try_object(entries);
  ASSERT_TRUE(root);
  auto source_result = Document::try_from_value(*root);
  ASSERT_TRUE(source_result);
  const Document source = std::move(*source_result);
  EXPECT_FALSE(source.digest().source.has_value());
  EXPECT_FALSE(source.digest().base_source.has_value());
  auto name_path = Path::try_parse("/name");
  ASSERT_TRUE(name_path);
  auto after = ConfigValue::try_string("after");
  ASSERT_TRUE(after);
  const Edit replace = Edit::replace(*name_path, *after);
  auto updated = source.try_patched(replace);
  ASSERT_TRUE(updated);
  // Key assertion: source is unchanged after the patch; updated carries the new value.
  EXPECT_EQ(source.try_at("/name")->try_as_string().value(), "before");
  EXPECT_EQ(updated->try_at("/name")->try_as_string().value(), "after");
  EXPECT_NE(source.digest().semantic.sha256, updated->digest().semantic.sha256);
  EXPECT_FALSE(updated->digest().source.has_value());
  EXPECT_FALSE(updated->digest().base_source.has_value());

  auto unchanged = source.try_patched({});
  ASSERT_TRUE(unchanged);
  EXPECT_EQ(unchanged->digest().semantic, source.digest().semantic);
  EXPECT_EQ(unchanged->digest().source, source.digest().source);
  EXPECT_EQ(unchanged->digest().base_source, source.digest().base_source);

  auto child_path = Path::try_parse("/name/child");
  ASSERT_TRUE(child_path);
  std::array overlap{replace, Edit::remove(*child_path)};
  auto failed = source.try_patched(overlap);
  // Key assertion: overlapping edits (replace /name then remove /name/child) fail with
  //               kPathConflict and leave source unchanged.
  EXPECT_FALSE(failed);
  EXPECT_EQ(failed.error().code(), ConfigErrorCode::kPathConflict);
  EXPECT_EQ(source.try_at("/name")->try_as_string().value(), "before");
}

TEST(ConfigDocument, SemanticEncodingIsTypedAndObjectOrderIndependent) {
  // REQ: CONFIG-004
  // Key setup: two objects with the same key-value pairs in different insertion order.
  auto signed_one = ConfigValue::try_signed_integer(1);
  auto unsigned_one = ConfigValue::try_unsigned_integer(1);
  ASSERT_TRUE(signed_one);
  ASSERT_TRUE(unsigned_one);
  std::array first_entries{ConfigObjectEntry{"z", *signed_one},
                           ConfigObjectEntry{"a", *unsigned_one}};
  std::array second_entries{ConfigObjectEntry{"a", *unsigned_one},
                            ConfigObjectEntry{"z", *signed_one}};
  auto first = ConfigValue::try_object(first_entries);
  auto second = ConfigValue::try_object(second_entries);
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  auto first_bytes = try_encode_semantic_value_v1(first->view());
  auto second_bytes = try_encode_semantic_value_v1(second->view());
  ASSERT_TRUE(first_bytes);
  ASSERT_TRUE(second_bytes);
  EXPECT_EQ(*first_bytes, *second_bytes);
  auto first_document = Document::try_from_value(*first);
  auto second_document = Document::try_from_value(*second);
  auto signed_document = Document::try_from_value(*signed_one);
  auto unsigned_document = Document::try_from_value(*unsigned_one);
  ASSERT_TRUE(first_document);
  ASSERT_TRUE(second_document);
  ASSERT_TRUE(signed_document);
  ASSERT_TRUE(unsigned_document);
  // Key assertion: objects with identical content but different key order share the same
  //               semantic digest; i64 vs u64 representation of "1" differs.
  EXPECT_EQ(first_document->digest().semantic.sha256, second_document->digest().semantic.sha256);
  EXPECT_NE(signed_document->digest().semantic.sha256, unsigned_document->digest().semantic.sha256);

  {
    const holoscan::test::ScopedAllocationFailure fail_next_allocation{0U};
    auto result = try_encode_semantic_value_v1(first->view());
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), ConfigErrorCode::kLimitExceeded);
  }

  try {
    const holoscan::test::ScopedAllocationFailure fail_next_allocation{0U};
    static_cast<void>(encode_semantic_value_v1(first->view()));
    FAIL() << "expected ConfigException";
  } catch (const ConfigException& exception) {
    EXPECT_EQ(exception.error().code(), ConfigErrorCode::kLimitExceeded);
  }
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

TEST(ConfigDocument, FormatsErrorsForHumanReadableLogs) {
  const std::array sensitive_excerpt{std::byte{'s'},
                                     std::byte{'e'},
                                     std::byte{'c'},
                                     std::byte{'r'},
                                     std::byte{'e'},
                                     std::byte{'t'}};
  const ConfigError cause =
      ConfigError::make(ConfigErrorCode::kTypeMismatch, "expected integer", "/camera/fps");
  const ConfigError error = ConfigError::nest(ConfigError::make(ConfigErrorCode::kParseError,
                                                                "could not\nparse",
                                                                "/camera/\x1b[2Jconfig",
                                                                "robot\r.yaml",
                                                                sensitive_excerpt),
                                              cause);

  const std::string formatted = format_error(error);

  EXPECT_EQ(formatted,
            "CONFIG_PARSE_ERROR: could not parse [source: robot .yaml] "
            "[path: /camera/ [2Jconfig]; caused by: CONFIG_TYPE_MISMATCH: "
            "expected integer [path: /camera/fps]");
}

TEST(ConfigDocument, FormatsTokenWhenNoOptionalContextExists) {
  const ConfigError error = ConfigError::make(ConfigErrorCode::kPathNotFound);

  const std::string formatted = format_error(error);

  EXPECT_EQ(formatted, "CONFIG_PATH_NOT_FOUND");
}

TEST(ConfigDocument, MarksTruncatedContext) {
  const std::string message(ConfigError::kMaxMessageBytes + 1U, 'm');
  const std::string path(ConfigError::kMaxPathBytes + 1U, 'p');
  const std::string source(ConfigError::kMaxSourceBytes + 1U, 's');
  const ConfigError error = ConfigError::make(ConfigErrorCode::kParseError, message, path, source);

  const std::string formatted = format_error(error);

  EXPECT_NE(formatted.find(std::string(ConfigError::kMaxMessageBytes, 'm') + "..."),
            std::string::npos);
  EXPECT_NE(formatted.find(std::string(ConfigError::kMaxPathBytes, 'p') + "..."),
            std::string::npos);
  EXPECT_NE(formatted.find(std::string(ConfigError::kMaxSourceBytes, 's') + "..."),
            std::string::npos);
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
