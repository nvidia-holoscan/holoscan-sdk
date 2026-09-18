/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/parameter_materializer.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "../common/allocation_counter.hpp"

namespace holoscan::authoring::test {

[[nodiscard]] bool even_value(const std::uint32_t& value) noexcept {
  return value % 2U == 0U;
}

// Exercises every scalar, string, duration, and validation path supported by the materializer.
// Representative test: AuthoringParameterMaterializer.CoercesNamedValuesAndPreservesDefaults.
struct MaterializerParams {
  std::int32_t count{1};
  std::uint32_t mode{2};
  float gain{1.0F};
  std::string label{"default"};
  std::chrono::milliseconds timeout{1};
  std::uint32_t even{2};
  std::vector<std::int32_t> samples{};
};

HOLOSCAN_PARAM_SCHEMA(
    MaterializerParams, holoscan::param_schema("com.nvidia.holoscan.test.MaterializerParams", 1),
    holoscan::field<&MaterializerParams::count>("count").range(-4, 4),
    holoscan::field<&MaterializerParams::mode>("mode").allowed_values({1U, 2U, 4U}),
    holoscan::field<&MaterializerParams::gain>("gain").range(0.0F, 2.0F),
    holoscan::field<&MaterializerParams::label>("label")
        .encoding(ParamStringEncoding::kUtf8)
        .max_utf8_bytes(12),
    holoscan::field<&MaterializerParams::timeout>("timeout").range(std::chrono::milliseconds{1},
                                                                   std::chrono::milliseconds{10}),
    holoscan::field<&MaterializerParams::even>("even").validator<even_value>(
        "test.materializer.even/v1"),
    holoscan::field<&MaterializerParams::samples>("samples"));

// Provides a distinct aggregate/schema pair for mismatch rejection.
// Representative test: AuthoringParameterMaterializer.RejectsMismatchedSchemaBeforeAllocation.
struct OtherParams {
  std::uint64_t value{};
};

HOLOSCAN_PARAM_SCHEMA(OtherParams,
                      holoscan::param_schema("com.nvidia.holoscan.test.OtherParams", 1),
                      holoscan::field<&OtherParams::value>("value"));

namespace {

[[nodiscard]] expected<MaterializedParameters, Error> materialize(
    std::span<const ParameterAssignment> assignments) {
  return materialize_parameters(param_type_descriptor<MaterializerParams>(),
                                param_schema_descriptor<MaterializerParams>(),
                                assignments);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, CoercesNamedValuesAndPreservesDefaults) {
  // Key setup: Supply cross-width scalar values for a subset of the parameter aggregate.
  const std::array assignments{
      ParameterAssignment{"count", ParameterValue::from(std::int64_t{-3})},
      ParameterAssignment{"gain", ParameterValue::from(1.5)},
      ParameterAssignment{"label", ParameterValue::from(std::string_view{"camera"})},
      ParameterAssignment{"timeout", ParameterValue::from(std::int64_t{5'000'000})},
  };
  auto materialized = materialize(assignments);

  // Key assertion: Values are coerced while fields without assignments retain their defaults.
  ASSERT_TRUE(materialized) << materialized.error().message;
  const auto& params = *static_cast<const MaterializerParams*>(materialized->data());
  EXPECT_EQ(params.count, -3);
  EXPECT_FLOAT_EQ(params.gain, 1.5F);
  EXPECT_EQ(params.label, "camera");
  EXPECT_EQ(params.timeout, std::chrono::milliseconds{5});
  EXPECT_EQ(params.mode, 2U);
  EXPECT_EQ(params.even, 2U);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, ReportsRangeAllowedAndCustomValidationWithCoreToken) {
  // Key setup: Exercise range, allowed-value, and custom-validator failures through one table.
  struct Case {
    ParameterAssignment assignment;
    std::string_view category;
    std::string_view validator;
  };
  const std::array cases{
      Case{ParameterAssignment{"count", ParameterValue::from(std::int64_t{7})},
           "category=out_of_range",
           ""},
      Case{ParameterAssignment{"mode", ParameterValue::from(std::uint64_t{3})},
           "category=not_allowed",
           ""},
      Case{ParameterAssignment{"even", ParameterValue::from(std::uint64_t{3})},
           "category=validator_rejected",
           "validator_id=test.materializer.even/v1"},
  };

  // Key assertion: Every validation route retains the shared token and its specific category.
  for (const auto& test_case : cases) {
    auto result = materialize(std::span{&test_case.assignment, 1U});
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("PARAM_VALIDATION_FAILED"), std::string::npos);
    EXPECT_NE(result.error().message.find(test_case.category), std::string::npos);
    if (!test_case.validator.empty()) {
      EXPECT_NE(result.error().message.find(test_case.validator), std::string::npos);
    }
  }
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, RejectsUnknownDuplicateAndMismatchedAssignmentsAsData) {
  // Key setup: Submit unknown, duplicate, and type-incompatible named assignments.
  const std::array unknown{ParameterAssignment{"missing", ParameterValue::from(std::int64_t{1})}};
  auto unknown_result = materialize(unknown);

  // Key assertion: Each authoring error is returned as data with its stable diagnostic token.
  ASSERT_FALSE(unknown_result);
  EXPECT_NE(unknown_result.error().message.find("AUTHORING_PARAMETER_UNKNOWN"), std::string::npos);

  const std::array duplicate{
      ParameterAssignment{"count", ParameterValue::from(std::int64_t{1})},
      ParameterAssignment{"count", ParameterValue::from(std::int64_t{2})},
  };
  auto duplicate_result = materialize(duplicate);
  ASSERT_FALSE(duplicate_result);
  EXPECT_NE(duplicate_result.error().message.find("AUTHORING_PARAMETER_DUPLICATE"),
            std::string::npos);

  const std::array mismatch{
      ParameterAssignment{"count", ParameterValue::from(std::string_view{"not-an-integer"})}};
  auto mismatch_result = materialize(mismatch);
  ASSERT_FALSE(mismatch_result);
  EXPECT_NE(mismatch_result.error().message.find("AUTHORING_PARAMETER_TYPE_MISMATCH"),
            std::string::npos);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, RejectsAmbiguousAliasesIndependentOfDeclarationOrder) {
  // Key setup: Build manual descriptors whose canonical and alias names collide.
  struct AliasParams {
    std::uint32_t first{};
    std::uint32_t second{};
  };
  static constexpr int kTypeToken{};
  const ParamTypeDescriptor type{
      .local_type_token = &kTypeToken,
      .object_size = sizeof(AliasParams),
      .object_alignment = alignof(AliasParams),
      .default_construct = [](void* value) { ::new (value) AliasParams{}; },
      .destroy = [](void* value) noexcept { static_cast<AliasParams*>(value)->~AliasParams(); },
      .validate_aggregate = [](const void*) noexcept { return ParamValidationResult{}; },
  };
  const auto field = [](std::string_view canonical_name, std::string_view alias) {
    return ParamFieldDescriptor{
        .canonical_name = canonical_name,
        .logical_type = ParamLogicalType::kUnsignedInteger,
        .type_shape = ParamFieldTypeShape{.type = ParamLogicalType::kUnsignedInteger,
                                          .bit_width = 32U,
                                          .has_stable_shape = true},
        .alias = alias,
        .local_operations =
            ParamFieldLocalOperations{
                .replace_member_from_scalar = [](void*, ParamScalarValue) noexcept { return true; },
            },
    };
  };
  const std::array assignment{
      ParameterAssignment{"shared", ParameterValue::from(std::uint32_t{1U})}};
  const std::array canonical_alias{
      field("first", "shared"),
      field("shared", ""),
  };
  const std::array alias_alias{
      field("first", "shared"),
      field("second", "shared"),
  };
  const std::array canonical_alias_reversed{canonical_alias[1], canonical_alias[0]};
  const std::array alias_alias_reversed{alias_alias[1], alias_alias[0]};

  const auto expect_ambiguous = [&](std::span<const ParamFieldDescriptor> fields) {
    const ParamSchemaDescriptor schema{.local_type = &type, .fields = fields};
    const auto result = materialize_parameters(type, &schema, assignment);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
    EXPECT_NE(result.error().message.find("AUTHORING_PARAMETER_AMBIGUOUS"), std::string::npos);
  };

  // Key assertion: Canonical/alias and alias/alias conflicts fail in both declaration orders.
  expect_ambiguous(canonical_alias);
  expect_ambiguous(canonical_alias_reversed);
  expect_ambiguous(alias_alias);
  expect_ambiguous(alias_alias_reversed);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, AcceptsUniqueAliasAndRejectsDuplicateAliasAssignment) {
  // Key setup: Give one scalar field a unique legacy alias.
  struct AliasParams {
    std::uint32_t value{};
  };
  static constexpr int kTypeToken{};
  const ParamTypeDescriptor type{
      .local_type_token = &kTypeToken,
      .object_size = sizeof(AliasParams),
      .object_alignment = alignof(AliasParams),
      .default_construct = [](void* value) { ::new (value) AliasParams{}; },
      .destroy = [](void* value) noexcept { static_cast<AliasParams*>(value)->~AliasParams(); },
      .validate_aggregate = [](const void*) noexcept { return ParamValidationResult{}; },
  };
  const std::array fields{ParamFieldDescriptor{
      .canonical_name = "current",
      .logical_type = ParamLogicalType::kUnsignedInteger,
      .type_shape = ParamFieldTypeShape{.type = ParamLogicalType::kUnsignedInteger,
                                        .bit_width = 32U,
                                        .has_stable_shape = true},
      .alias = "legacy",
      .local_operations =
          ParamFieldLocalOperations{
              .replace_member_from_scalar =
                  [](void* aggregate, ParamScalarValue scalar) noexcept {
                    static_cast<AliasParams*>(aggregate)->value =
                        static_cast<std::uint32_t>(scalar.raw_bits);
                    return true;
                  },
          },
  }};
  const ParamSchemaDescriptor schema{.local_type = &type, .fields = fields};
  const std::array unique{ParameterAssignment{"legacy", ParameterValue::from(std::uint32_t{7U})}};
  const auto accepted = materialize_parameters(type, &schema, unique);

  // Key assertion: A unique alias resolves normally, but alias-plus-canonical is one duplicate.
  ASSERT_TRUE(accepted);
  EXPECT_EQ(static_cast<const AliasParams*>(accepted->data())->value, 7U);
  const std::array duplicate{
      ParameterAssignment{"legacy", ParameterValue::from(std::uint32_t{7U})},
      ParameterAssignment{"current", ParameterValue::from(std::uint32_t{8U})},
  };
  const auto rejected = materialize_parameters(type, &schema, duplicate);
  ASSERT_FALSE(rejected);
  EXPECT_NE(rejected.error().message.find("AUTHORING_PARAMETER_DUPLICATE"), std::string::npos);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, ValidatesStringAndDurationAfterReplacement) {
  // Key setup: Materialize values that pass coercion but violate their declared field bounds.
  const std::array invalid_string{
      ParameterAssignment{"label", ParameterValue::from(std::string_view{"too-many-bytes"})}};
  auto string_result = materialize(invalid_string);

  // Key assertion: Post-replacement validation identifies the failing field without its value.
  ASSERT_FALSE(string_result);
  EXPECT_NE(string_result.error().message.find("PARAM_VALIDATION_FAILED"), std::string::npos);
  EXPECT_NE(string_result.error().message.find("field_name=label"), std::string::npos);

  const std::array invalid_duration{
      ParameterAssignment{"timeout", ParameterValue::from(std::int64_t{11'000'000})}};
  auto duration_result = materialize(invalid_duration);
  ASSERT_FALSE(duration_result);
  EXPECT_NE(duration_result.error().message.find("PARAM_VALIDATION_FAILED"), std::string::npos);
  EXPECT_NE(duration_result.error().message.find("field_name=timeout"), std::string::npos);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, RejectsMismatchedSchemaBeforeAllocation) {
  // Key setup: Pair one aggregate descriptor with another aggregate's schema.
  const auto result = materialize_parameters(
      param_type_descriptor<MaterializerParams>(), param_schema_descriptor<OtherParams>(), {});

  // Key assertion: The unsafe descriptor pairing is rejected before any field thunk can run.
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_NE(result.error().message.find("AUTHORING_PARAMETER_SCHEMA_MISMATCH"), std::string::npos);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, DistinguishesStagedKindsFromTypeMismatch) {
  // Key setup: Address a declared sequence through the initial scalar/string-only value carrier.
  const std::array assignments{
      ParameterAssignment{"samples", ParameterValue::from(std::string_view{"unsupported"})}};
  const auto result = materialize(assignments);

  // Key assertion: Bindings can distinguish a staged field kind from a malformed scalar value.
  ASSERT_FALSE(result);
  EXPECT_NE(result.error().message.find("AUTHORING_PARAMETER_KIND_UNSUPPORTED"), std::string::npos);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, ConvertsMaterializationExceptionsToFailureData) {
  // Key setup: Supply an otherwise valid descriptor whose aggregate construction throws.
  static constexpr int kTypeToken{};
  const ParamTypeDescriptor throwing_type{
      .local_type_token = &kTypeToken,
      .object_size = sizeof(std::uint32_t),
      .object_alignment = alignof(std::uint32_t),
      .default_construct = [](void*) { throw std::runtime_error{"expected"}; },
      .destroy = [](void*) noexcept {},
      .validate_aggregate = [](const void*) noexcept { return ParamValidationResult{}; },
  };

  // Key assertion: No exception escapes the language-neutral materialization boundary.
  const auto result = materialize_parameters(throwing_type, nullptr, {});
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kFailure);
  EXPECT_NE(result.error().message.find("AUTHORING_PARAMETER_CONSTRUCTION_FAILED"),
            std::string::npos);

  struct ThrowingAssignmentParams {};
  const ParamTypeDescriptor assignment_type{
      .local_type_token = &kTypeToken,
      .object_size = sizeof(ThrowingAssignmentParams),
      .object_alignment = alignof(ThrowingAssignmentParams),
      .default_construct =
          [](void* destination) { ::new (destination) ThrowingAssignmentParams{}; },
      .destroy =
          [](void* value) noexcept {
            static_cast<ThrowingAssignmentParams*>(value)->~ThrowingAssignmentParams();
          },
      .validate_aggregate = [](const void*) noexcept { return ParamValidationResult{}; },
  };
  const std::array fields{ParamFieldDescriptor{
      .canonical_name = "value",
      .logical_type = ParamLogicalType::kString,
      .local_operations =
          ParamFieldLocalOperations{
              .replace_member_from_string =
                  [](void*, std::string_view) { throw std::runtime_error{"expected"}; },
          },
  }};
  const ParamSchemaDescriptor assignment_schema{
      .local_type = &assignment_type,
      .fields = fields,
  };
  const std::array assignments{
      ParameterAssignment{"value", ParameterValue::from(std::string_view{"assigned"})}};

  // Key assertion: Field-assignment exceptions are translated through the same failure boundary.
  const auto assignment_result =
      materialize_parameters(assignment_type, &assignment_schema, assignments);
  ASSERT_FALSE(assignment_result);
  EXPECT_EQ(assignment_result.error().code, ErrorCode::kFailure);
  EXPECT_NE(assignment_result.error().message.find("AUTHORING_PARAMETER_ASSIGNMENT_FAILED"),
            std::string::npos);
}

// REQ: AUTHOR-003
TEST(AuthoringParameterMaterializer, KeepsAllocationFailureFallbackExceptionFree) {
  // Key setup: Make the field operation itself report allocation exhaustion.
  struct ThrowingAssignmentParams {};
  static constexpr int kTypeToken{};
  const ParamTypeDescriptor type{
      .local_type_token = &kTypeToken,
      .object_size = sizeof(ThrowingAssignmentParams),
      .object_alignment = alignof(ThrowingAssignmentParams),
      .default_construct = [](void* value) { ::new (value) ThrowingAssignmentParams{}; },
      .destroy =
          [](void* value) noexcept {
            static_cast<ThrowingAssignmentParams*>(value)->~ThrowingAssignmentParams();
          },
      .validate_aggregate = [](const void*) noexcept { return ParamValidationResult{}; },
  };
  const std::array fields{ParamFieldDescriptor{
      .canonical_name = "value",
      .logical_type = ParamLogicalType::kString,
      .local_operations =
          ParamFieldLocalOperations{
              .replace_member_from_string = [](void*, std::string_view) { throw std::bad_alloc{}; },
          },
  }};
  const ParamSchemaDescriptor schema{.local_type = &type, .fields = fields};
  const std::array assignments{
      ParameterAssignment{"value", ParameterValue::from(std::string_view{"assigned"})}};
  std::size_t allocation_count{};
  {
    holoscan::test::ScopedAllocationCounter counter;
    const auto measured = materialize_parameters(type, &schema, assignments);
    ASSERT_FALSE(measured);
    allocation_count = counter.count();
  }
  ASSERT_GT(allocation_count, 0U);

  // Key assertion: Failure of the final rich diagnostic returns code-only data without escaping.
  holoscan::test::ScopedAllocationFailure failure{allocation_count - 1U};
  const auto result = materialize_parameters(type, &schema, assignments);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kResourceExhausted);
  EXPECT_TRUE(result.error().message.empty());
}

}  // namespace
}  // namespace holoscan::authoring::test
