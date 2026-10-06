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
#include <filesystem>
#include <new>
#include <optional>
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

// Supplies named enum values exercised by structured materialization.
// Representative test: AuthoringParameterMaterializer.MaterializesStructuredValues.
enum class MaterializerMode : std::uint8_t {
  kDisabled = 0,
  kAutomatic = 2,
  kManual = 7,
};

// Defines the leaf aggregate used to verify recursive assignment.
// Representative test: AuthoringParameterMaterializer.MaterializesTwoLevelNestedAssignmentGroups.
struct SensorSettings {
  std::uint32_t channels{1U};
};

HOLOSCAN_PARAM_SCHEMA(SensorSettings,
                      holoscan::param_schema("com.nvidia.holoscan.test.SensorSettings", 1),
                      holoscan::field<&SensorSettings::channels>("channels").range(1U, 4U));

// Defines an intermediate nested aggregate used to verify recursive traversal.
// Representative test: AuthoringParameterMaterializer.ReportsCanonicalFullNestedPaths.
struct CameraSettings {
  SensorSettings sensor{};
};

HOLOSCAN_PARAM_SCHEMA(CameraSettings,
                      holoscan::param_schema("com.nvidia.holoscan.test.CameraSettings", 1),
                      holoscan::field<&CameraSettings::sensor>("sensor").nested<SensorSettings>());

// Reuses one nested schema at two paths to verify value-aware diagnostic-path reconstruction.
// Representative test: AuthoringParameterMaterializer.ResolvesRepeatedNestedSchemaValidationPath.
struct RepeatedNestedParams {
  std::optional<SensorSettings> first{};
  SensorSettings second{7U};
};

HOLOSCAN_PARAM_SCHEMA(
    RepeatedNestedParams,
    holoscan::param_schema("com.nvidia.holoscan.test.RepeatedNestedParams", 1),
    holoscan::field<&RepeatedNestedParams::first>("first").nested<SensorSettings>(),
    holoscan::field<&RepeatedNestedParams::second>("second").nested<SensorSettings>());

// Collects the supported root-level field shapes for materializer tests.
// Representative test: AuthoringParameterMaterializer.MaterializesStructuredValues.
struct MaterializerParams {
  std::int32_t count{1};
  std::uint32_t mode{2};
  float gain{1.0F};
  std::string label{"default"};
  std::chrono::milliseconds timeout{1};
  std::uint32_t even{2};
  std::vector<std::int32_t> levels{1};
  std::vector<std::string> labels{"default"};
  std::filesystem::path output_path{"default"};
  std::optional<std::uint32_t> retained{7U};
  std::optional<std::uint32_t> cleared{8U};
  MaterializerMode enum_mode{MaterializerMode::kAutomatic};
  std::optional<MaterializerMode> fallback{MaterializerMode::kAutomatic};
  std::vector<MaterializerMode> modes{MaterializerMode::kDisabled};
  CameraSettings camera{};
  std::optional<CameraSettings> backup_camera{};
  CameraSettings opaque_camera{};
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
    holoscan::field<&MaterializerParams::levels>("levels").max_elements(4),
    holoscan::field<&MaterializerParams::labels>("labels")
        .element_encoding(ParamStringEncoding::kUtf8)
        .max_element_utf8_bytes(8)
        .max_elements(3),
    holoscan::field<&MaterializerParams::output_path>("output_path"),
    holoscan::field<&MaterializerParams::retained>("retained"),
    holoscan::field<&MaterializerParams::cleared>("cleared"),
    holoscan::field<&MaterializerParams::enum_mode>("enum_mode")
        .enum_values({param_enum_value("disabled", MaterializerMode::kDisabled),
                      param_enum_value("automatic", MaterializerMode::kAutomatic),
                      param_enum_value("manual", MaterializerMode::kManual)}),
    holoscan::field<&MaterializerParams::fallback>("fallback")
        .enum_values({param_enum_value("disabled", MaterializerMode::kDisabled),
                      param_enum_value("automatic", MaterializerMode::kAutomatic),
                      param_enum_value("manual", MaterializerMode::kManual)}),
    holoscan::field<&MaterializerParams::modes>("modes")
        .enum_values({param_enum_value("disabled", MaterializerMode::kDisabled),
                      param_enum_value("automatic", MaterializerMode::kAutomatic),
                      param_enum_value("manual", MaterializerMode::kManual)})
        .max_elements(4),
    holoscan::field<&MaterializerParams::camera>("camera").nested<CameraSettings>(),
    holoscan::field<&MaterializerParams::backup_camera>("backup_camera").nested<CameraSettings>(),
    holoscan::field<&MaterializerParams::opaque_camera>("opaque_camera"));

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
// REQ: AUTHOR-015
TEST(AuthoringParameterMaterializer, CoercesNamedValuesAndPreservesDefaults) {
  // Key setup: Mix signed, floating-point, string, and duration assignments with defaults.
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

// REQ: AUTHOR-018
// REQ: AUTHOR-015
TEST(AuthoringParameterMaterializer, MaterializesStructuredValues) {
  // Key setup: Assign sequences, a path, explicit absence, and enum names/values together.
  const std::array level_values{ParameterValue::from(std::int64_t{-2}),
                                ParameterValue::from(std::uint64_t{4})};
  const std::array label_values{ParameterValue::from(std::string_view{"rgb"}),
                                ParameterValue::from(std::string_view{"left"})};
  const std::array mode_values{ParameterValue::from(std::string_view{"automatic"}),
                               ParameterValue::from(std::uint64_t{7})};
  const std::array assignments{
      ParameterAssignment{"levels", ParameterValue::from(std::span{level_values})},
      ParameterAssignment{"labels", ParameterValue::from(std::span{label_values})},
      ParameterAssignment{"output_path", ParameterValue::from(std::string_view{"results/frame"})},
      ParameterAssignment{"cleared", ParameterValue::none()},
      ParameterAssignment{"enum_mode", ParameterValue::from(std::string_view{"manual"})},
      ParameterAssignment{"fallback", ParameterValue::from(std::uint64_t{0})},
      ParameterAssignment{"modes", ParameterValue::from(std::span{mode_values})},
  };
  auto materialized = materialize(assignments);

  // Key assertion: Structured values use schema-owned exact-type operations and preserve defaults.
  ASSERT_TRUE(materialized) << materialized.error().message;
  const auto& params = *static_cast<const MaterializerParams*>(materialized->data());
  EXPECT_EQ(params.levels, (std::vector<std::int32_t>{-2, 4}));
  EXPECT_EQ(params.labels, (std::vector<std::string>{"rgb", "left"}));
  EXPECT_EQ(params.output_path, std::filesystem::path{"results/frame"});
  ASSERT_EQ(params.retained, 7U);
  EXPECT_FALSE(params.cleared.has_value());
  EXPECT_EQ(params.enum_mode, MaterializerMode::kManual);
  ASSERT_EQ(params.fallback, MaterializerMode::kDisabled);
  EXPECT_EQ(
      params.modes,
      (std::vector<MaterializerMode>{MaterializerMode::kAutomatic, MaterializerMode::kManual}));
}

// REQ: AUTHOR-018
TEST(AuthoringParameterMaterializer, RejectsInvalidAbsenceAndSequenceElements) {
  // Key setup: Exercise unset, invalid absence, enum-label, and sequence forms independently.
  const std::array unset{ParameterAssignment{"cleared", ParameterValue{}}};
  auto unset_result = materialize(unset);
  ASSERT_FALSE(unset_result);
  EXPECT_NE(unset_result.error().message.find("AUTHORING_PARAMETER_VALUE_UNSET"),
            std::string::npos);

  const std::array invalid_none{ParameterAssignment{"count", ParameterValue::none()}};
  auto none_result = materialize(invalid_none);

  // Key assertion: Every incompatible structured value fails with stable field-aware diagnostics.
  ASSERT_FALSE(none_result);
  EXPECT_NE(none_result.error().message.find("AUTHORING_PARAMETER_TYPE_MISMATCH"),
            std::string::npos);
  EXPECT_NE(none_result.error().message.find("field_name=count"), std::string::npos);

  const std::array invalid_mode_values{ParameterValue::from(std::string_view{"not-a-mode"})};
  const std::array invalid_mode{
      ParameterAssignment{"modes", ParameterValue::from(std::span{invalid_mode_values})}};
  auto mode_result = materialize(invalid_mode);
  ASSERT_FALSE(mode_result);
  EXPECT_NE(mode_result.error().message.find("AUTHORING_PARAMETER_TYPE_MISMATCH"),
            std::string::npos);
  EXPECT_NE(mode_result.error().message.find("field_name=modes"), std::string::npos);

  const std::array wrong_label_values{ParameterValue::from(std::uint64_t{1})};
  const std::array wrong_labels{
      ParameterAssignment{"labels", ParameterValue::from(std::span{wrong_label_values})}};
  auto label_result = materialize(wrong_labels);
  ASSERT_FALSE(label_result);
  EXPECT_NE(label_result.error().message.find("AUTHORING_PARAMETER_TYPE_MISMATCH"),
            std::string::npos);

  const std::array too_many_values{ParameterValue::from(1),
                                   ParameterValue::from(2),
                                   ParameterValue::from(3),
                                   ParameterValue::from(4),
                                   ParameterValue::from(5)};
  const std::array too_many{
      ParameterAssignment{"levels", ParameterValue::from(std::span{too_many_values})}};
  auto cardinality_result = materialize(too_many);
  ASSERT_FALSE(cardinality_result);
  EXPECT_NE(cardinality_result.error().message.find("category=cardinality_exceeded"),
            std::string::npos);
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringParameterMaterializer, DistinguishesUnsupportedKindsFromTypeMismatch) {
  // Key setup: Compare a correct nested carrier lacking schema operations with a wrong scalar kind.
  const std::array nested_fields{
      ParameterAssignment{"sensor", ParameterValue::from(std::span<const ParameterAssignment>{})}};
  const std::array unsupported{
      ParameterAssignment{"opaque_camera", ParameterValue::from(std::span{nested_fields})}};
  const auto unsupported_result = materialize(unsupported);
  const std::array mismatch{
      ParameterAssignment{"count", ParameterValue::from(std::string_view{"wrong"})}};
  const auto mismatch_result = materialize(mismatch);

  // Key assertion: Capability gaps and caller type errors retain distinct stable tokens.
  ASSERT_FALSE(unsupported_result);
  EXPECT_NE(unsupported_result.error().message.find("AUTHORING_PARAMETER_KIND_UNSUPPORTED"),
            std::string::npos);
  ASSERT_FALSE(mismatch_result);
  EXPECT_NE(mismatch_result.error().message.find("AUTHORING_PARAMETER_TYPE_MISMATCH"),
            std::string::npos);
}

// REQ: AUTHOR-019
TEST(AuthoringParameterMaterializer, MaterializesTwoLevelNestedAssignmentGroups) {
  // Key setup: Build nested assignment groups for required and optional camera aggregates.
  const std::array sensor_assignments{
      ParameterAssignment{"channels", ParameterValue::from(std::uint64_t{3})}};
  const std::array camera_assignments{
      ParameterAssignment{"sensor", ParameterValue::from(std::span{sensor_assignments})}};
  const std::array backup_sensor_assignments{
      ParameterAssignment{"channels", ParameterValue::from(std::uint64_t{4})}};
  const std::array backup_camera_assignments{
      ParameterAssignment{"sensor", ParameterValue::from(std::span{backup_sensor_assignments})}};
  const std::array assignments{
      ParameterAssignment{"camera", ParameterValue::from(std::span{camera_assignments})},
      ParameterAssignment{"backup_camera",
                          ParameterValue::from(std::span{backup_camera_assignments})},
  };

  auto materialized = materialize(assignments);

  // Key assertion: Recursive traversal updates both direct and newly engaged optional children.
  ASSERT_TRUE(materialized) << materialized.error().message;
  const auto& params = *static_cast<const MaterializerParams*>(materialized->data());
  EXPECT_EQ(params.camera.sensor.channels, 3U);
  ASSERT_TRUE(params.backup_camera.has_value());
  if (params.backup_camera.has_value()) {
    EXPECT_EQ(params.backup_camera->sensor.channels, 4U);
  }
}

// REQ: AUTHOR-019
TEST(AuthoringParameterMaterializer, ReportsCanonicalFullNestedPaths) {
  // Key setup: Violate a leaf constraint below two nested aggregate fields.
  const std::array invalid_sensor{
      ParameterAssignment{"channels", ParameterValue::from(std::uint64_t{7})}};
  const std::array invalid_camera{
      ParameterAssignment{"sensor", ParameterValue::from(std::span{invalid_sensor})}};
  const std::array invalid_root{
      ParameterAssignment{"camera", ParameterValue::from(std::span{invalid_camera})}};
  auto invalid_result = materialize(invalid_root);

  // Key assertion: Validation diagnostics identify the complete canonical path to the leaf.
  ASSERT_FALSE(invalid_result);
  EXPECT_NE(invalid_result.error().message.find("PARAM_VALIDATION_FAILED"), std::string::npos);
  EXPECT_NE(invalid_result.error().message.find("field_name=camera.sensor.channels"),
            std::string::npos);

  const std::array invalid_backup_root{
      ParameterAssignment{"backup_camera", ParameterValue::from(std::span{invalid_camera})}};
  auto invalid_backup_result = materialize(invalid_backup_root);
  ASSERT_FALSE(invalid_backup_result);
  EXPECT_NE(invalid_backup_result.error().message.find("field_name=backup_camera.sensor.channels"),
            std::string::npos);

  const std::array unknown_sensor{
      ParameterAssignment{"missing", ParameterValue::from(std::uint64_t{1})}};
  const std::array unknown_camera{
      ParameterAssignment{"sensor", ParameterValue::from(std::span{unknown_sensor})}};
  const std::array unknown_root{
      ParameterAssignment{"camera", ParameterValue::from(std::span{unknown_camera})}};
  auto unknown_result = materialize(unknown_root);

  // Key assertion: Unknown nested names likewise retain the complete path from the root.
  ASSERT_FALSE(unknown_result);
  EXPECT_NE(unknown_result.error().message.find("AUTHORING_PARAMETER_UNKNOWN"), std::string::npos);
  EXPECT_NE(unknown_result.error().message.find("field_name=camera.sensor.missing"),
            std::string::npos);
}

// REQ: AUTHOR-019
TEST(AuthoringParameterMaterializer, ResolvesRepeatedNestedSchemaValidationPath) {
  // Key setup: The first instance of the shared nested schema is absent while the second instance
  // has an invalid default. Then repeat with the first instance present and valid.
  const std::span<const ParameterAssignment> no_assignments{};
  auto absent_first = materialize_parameters(param_type_descriptor<RepeatedNestedParams>(),
                                             param_schema_descriptor<RepeatedNestedParams>(),
                                             no_assignments);
  const std::array first_values{
      ParameterAssignment{"channels", ParameterValue::from(std::uint64_t{2})}};
  const std::array present_first_assignments{
      ParameterAssignment{"first", ParameterValue::from(std::span{first_values})}};
  auto valid_first = materialize_parameters(param_type_descriptor<RepeatedNestedParams>(),
                                            param_schema_descriptor<RepeatedNestedParams>(),
                                            present_first_assignments);

  // Key assertion: Reconstruction follows the failing value, not the first matching schema path.
  ASSERT_FALSE(absent_first);
  EXPECT_NE(absent_first.error().message.find("field_name=second.channels"), std::string::npos);
  ASSERT_FALSE(valid_first);
  EXPECT_NE(valid_first.error().message.find("field_name=second.channels"), std::string::npos);
}

// REQ: AUTHOR-003
// REQ: AUTHOR-015
TEST(AuthoringParameterMaterializer, ReportsRangeAllowedAndCustomValidationWithCoreToken) {
  // Key setup: Exercise range, allowed-value, and custom-validator rejection categories.
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

  for (const auto& test_case : cases) {
    auto result = materialize(std::span{&test_case.assignment, 1U});

    // Key assertion: Each rejected assignment retains the stable Core token and category.
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
// REQ: AUTHOR-015
TEST(AuthoringParameterMaterializer, RejectsUnknownDuplicateAndMismatchedAssignmentsAsData) {
  // Key setup: Materialize unknown, duplicate, and type-mismatched named assignments.
  const std::array unknown{ParameterAssignment{"missing", ParameterValue::from(std::int64_t{1})}};
  auto unknown_result = materialize(unknown);

  // Key assertion: Each authoring error remains a distinct non-throwing data result.
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
// REQ: AUTHOR-015
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
// REQ: AUTHOR-015
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
// REQ: AUTHOR-015
TEST(AuthoringParameterMaterializer, ValidatesStringAndDurationAfterReplacement) {
  // Key setup: Replace valid defaults with an oversized string and out-of-range duration.
  const std::array invalid_string{
      ParameterAssignment{"label", ParameterValue::from(std::string_view{"too-many-bytes"})}};
  auto string_result = materialize(invalid_string);

  // Key assertion: Validation runs against the reconstructed aggregate values.
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
// REQ: AUTHOR-015
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
// REQ: AUTHOR-015
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
// REQ: AUTHOR-015
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
    const holoscan::test::ScopedAllocationCounter counter;
    const auto measured = materialize_parameters(type, &schema, assignments);
    ASSERT_FALSE(measured);
    allocation_count = counter.count();
  }
  ASSERT_GT(allocation_count, 0U);

  // Key assertion: Failure of the final rich diagnostic returns code-only data without escaping.
  const holoscan::test::ScopedAllocationFailure failure{allocation_count - 1U};
  const auto result = materialize_parameters(type, &schema, assignments);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kResourceExhausted);
  EXPECT_TRUE(result.error().message.empty());
}

}  // namespace
}  // namespace holoscan::authoring::test
