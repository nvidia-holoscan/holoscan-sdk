/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/params/config.hpp>

#include "compiler/execution_plan_state.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/parameter_catalog_internal.hpp"
#include "core/parameter_input_internal.hpp"
#include "core/parameter_snapshot_archive_access.hpp"

namespace holoscan::params {

// Scalar-like element decoded inside ContainerBindingParams history vectors.
// Representative test: TypedBindingComposesVectorWithVisibleElementCodec.
struct ConfigBindingHistoryTerm {
  std::string input_name;
};

// Comparable aggregate member intentionally omitted by the aggregate-only codec.
// Representative test: ParamSchemaCodecsApplyFieldPresenceWithoutRepeatingFields.
struct ComparableOmittedField {
  std::int32_t value{7};
  bool operator==(const ComparableOmittedField&) const noexcept = default;
};

// Aggregate-codec carrier with one supported and one deliberately omitted field route.
// Representative test: GraphAttachmentRejectsAggregateOnlyFieldCodec.
struct AggregateCodecBindingParams {
  std::vector<std::string> labels{"baseline"};
  ComparableOmittedField untouched;
};

// Parameter carrier whose value changes an Operator input declaration.
// Representative test: CompileRejectsParameterDependentDeclarationChanges.
struct DeclarationBindingParams {
  std::int32_t input_index{};
};

// Nested inference settings with a canonical engine_name field and legacy engine alias.
// Representative test: RequiredNestedSchemaCodecNormalizesAliasDuringBinding.
struct SchemaCodecInferenceParams {
  std::int32_t iterations{2};
  std::string engine{"default"};

  bool operator==(const SchemaCodecInferenceParams&) const = default;
};

// Aggregate codec carrier combining required nested inference settings with a defaulted flag.
// Representative test: ParamSchemaCodecsApplyFieldPresenceWithoutRepeatingFields.
struct SchemaCodecParams {
  SchemaCodecInferenceParams inference;
  bool enabled{true};

  bool operator==(const SchemaCodecParams&) const = default;
};

// Flat C++ carrier whose schema fields materialize structured configuration paths.
// Representative test: SchemaDerivedCodecMaterializesStructuredCanonicalPaths.
struct StructuredSchemaCodecParams {
  std::int32_t iterations{2};
  double fine_gain{1.0};
  bool enabled{true};

  bool operator==(const StructuredSchemaCodecParams&) const = default;
};

// Defaulted nested settings used to contrast missing-value and unknown-field policies.
// Representative test: DefaultedNestedSchemaCodecNormalizesAliasUnderBindingPolicies.
struct DefaultedSchemaCodecInferenceParams {
  std::string engine{"default"};
  bool enabled{true};

  bool operator==(const DefaultedSchemaCodecInferenceParams&) const = default;
};

// Carrier with literal keys where one field name is a strict prefix of another.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForStrictPrefixOverlap.
struct PrefixOverlapSchemaCodecParams {
  std::int32_t tuning{2};
  std::int32_t iterations{3};

  bool operator==(const PrefixOverlapSchemaCodecParams&) const = default;
};

// Carrier with distinct literal keys that decode to the same path spelling.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForDecodedEqualPaths.
struct DecodedEqualSchemaCodecParams {
  std::int32_t plain{2};
  std::int32_t escaped{3};

  bool operator==(const DecodedEqualSchemaCodecParams&) const = default;
};

// Nested carrier whose alias is a prefix of another field's canonical path.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForAliasCanonicalPrefixOverlap.
struct AliasCanonicalPrefixSchemaCodecParams {
  std::int32_t aliased{2};
  std::int32_t descendant{3};

  bool operator==(const AliasCanonicalPrefixSchemaCodecParams&) const = default;
};

// Operator parameter wrapper for AliasCanonicalPrefixSchemaCodecParams.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForAliasCanonicalPrefixOverlap.
struct AliasCanonicalPrefixBindingParams {
  AliasCanonicalPrefixSchemaCodecParams value;
};

// One structured-codec child used to consume a shared encode/decode field budget.
// Representative test: StructuredSchemaCodecSharesBudgetAndRemapsChildErrorPaths.
struct StructuredBudgetValue {
  std::int32_t value{};

  bool operator==(const StructuredBudgetValue&) const = default;
};

// Two-child carrier that deliberately exceeds its shared structured-codec budget.
// Representative test: StructuredSchemaCodecSharesBudgetAndRemapsChildErrorPaths.
struct StructuredBudgetSchemaCodecParams {
  StructuredBudgetValue first;
  StructuredBudgetValue second;

  bool operator==(const StructuredBudgetSchemaCodecParams&) const = default;
};

// Handwritten codec carrier used to prove schema aliases are not inferred automatically.
// Representative test: HandwrittenNestedCodecDoesNotNormalizeSchemaAliases.
struct HandwrittenAliasParams {
  std::string engine{"default"};

  bool operator==(const HandwrittenAliasParams&) const = default;
};

// Operator parameter wrapper for handwritten nested decoding and baseline retention.
// Representative test: HandwrittenNestedCodecDoesNotNormalizeSchemaAliases.
struct HandwrittenNestedAliasBindingParams {
  HandwrittenAliasParams handwritten;
  std::int32_t untouched{7};
};

HOLOSCAN_PARAM_SCHEMA(AggregateCodecBindingParams,
                      param_schema("holoscan.test.AggregateCodecBindingParams", 1),
                      field<&AggregateCodecBindingParams::labels>("labels"),
                      field<&AggregateCodecBindingParams::untouched>("untouched"));

HOLOSCAN_PARAM_SCHEMA(DeclarationBindingParams,
                      param_schema("holoscan.test.DeclarationBindingParams", 1),
                      field<&DeclarationBindingParams::input_index>("input_index"));

HOLOSCAN_PARAM_SCHEMA(SchemaCodecInferenceParams,
                      param_schema("holoscan.test.SchemaCodecInferenceParams", 1),
                      field<&SchemaCodecInferenceParams::iterations>("iterations"),
                      field<&SchemaCodecInferenceParams::engine>("engine_name").alias("engine"));

HOLOSCAN_PARAM_SCHEMA(
    SchemaCodecParams, param_schema("holoscan.test.SchemaCodecParams", 1),
    field<&SchemaCodecParams::inference>("inference").nested<SchemaCodecInferenceParams>(),
    field<&SchemaCodecParams::enabled>("enabled"));

HOLOSCAN_PARAM_SCHEMA(StructuredSchemaCodecParams,
                      param_schema("holoscan.test.StructuredSchemaCodecParams", 1),
                      field<&StructuredSchemaCodecParams::iterations>("tuning/iterations"),
                      field<&StructuredSchemaCodecParams::fine_gain>("tuning/gain%2Ffine"),
                      field<&StructuredSchemaCodecParams::enabled>("enabled"));

HOLOSCAN_PARAM_SCHEMA(
    DefaultedSchemaCodecInferenceParams,
    param_schema("holoscan.test.DefaultedSchemaCodecInferenceParams", 1),
    field<&DefaultedSchemaCodecInferenceParams::engine>("engine_name").alias("engine"),
    field<&DefaultedSchemaCodecInferenceParams::enabled>("enabled"));

HOLOSCAN_PARAM_SCHEMA(PrefixOverlapSchemaCodecParams,
                      param_schema("holoscan.test.PrefixOverlapSchemaCodecParams", 1),
                      field<&PrefixOverlapSchemaCodecParams::tuning>("tuning"),
                      field<&PrefixOverlapSchemaCodecParams::iterations>("tuning/iterations"));

HOLOSCAN_PARAM_SCHEMA(DecodedEqualSchemaCodecParams,
                      param_schema("holoscan.test.DecodedEqualSchemaCodecParams", 1),
                      field<&DecodedEqualSchemaCodecParams::plain>("foo"),
                      field<&DecodedEqualSchemaCodecParams::escaped>("%66oo"));

HOLOSCAN_PARAM_SCHEMA(AliasCanonicalPrefixSchemaCodecParams,
                      param_schema("holoscan.test.AliasCanonicalPrefixSchemaCodecParams", 1),
                      field<&AliasCanonicalPrefixSchemaCodecParams::aliased>("alpha").alias("foo"),
                      field<&AliasCanonicalPrefixSchemaCodecParams::descendant>("foo/bar"));

HOLOSCAN_PARAM_SCHEMA(AliasCanonicalPrefixBindingParams,
                      param_schema("holoscan.test.AliasCanonicalPrefixBindingParams", 1),
                      field<&AliasCanonicalPrefixBindingParams::value>("value")
                          .nested<AliasCanonicalPrefixSchemaCodecParams>());

HOLOSCAN_PARAM_SCHEMA(StructuredBudgetSchemaCodecParams,
                      param_schema("holoscan.test.StructuredBudgetSchemaCodecParams", 1),
                      field<&StructuredBudgetSchemaCodecParams::first>("budget/first"),
                      field<&StructuredBudgetSchemaCodecParams::second>("budget/second"));

HOLOSCAN_PARAM_SCHEMA(HandwrittenAliasParams,
                      param_schema("holoscan.test.HandwrittenAliasParams", 1),
                      field<&HandwrittenAliasParams::engine>("engine_name").alias("engine"));

HOLOSCAN_PARAM_SCHEMA(HandwrittenNestedAliasBindingParams,
                      param_schema("holoscan.test.HandwrittenNestedAliasBindingParams", 1),
                      field<&HandwrittenNestedAliasBindingParams::handwritten>("handwritten")
                          .nested<HandwrittenAliasParams>(),
                      field<&HandwrittenNestedAliasBindingParams::untouched>("untouched"));

}  // namespace holoscan::params

// Scalar codec used by vector composition without an aggregate parameter codec.
// Representative test: TypedBindingComposesVectorWithVisibleElementCodec.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::ConfigBindingHistoryTerm> {
  static ConfigResult<holoscan::params::ConfigBindingHistoryTerm> decode(
      ConfigView view, const DecodeOptions& options, DecodeContext& context) {
    auto input_name = ConfigCodec<std::string>::decode(std::move(view), options, context);
    if (!input_name) {
      return tl::make_unexpected(std::move(input_name.error()));
    }
    return holoscan::params::ConfigBindingHistoryTerm{.input_name = std::move(*input_name)};
  }

  static ConfigResult<ConfigValue> encode(const holoscan::params::ConfigBindingHistoryTerm& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    return ConfigCodec<std::string>::encode(value.input_name, options, context);
  }
};

// Aggregate-only codec used to contrast aggregate decoding with field-route availability.
// Representative test: GraphAttachmentRejectsAggregateOnlyFieldCodec.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::AggregateCodecBindingParams> {
  static ConfigResult<holoscan::params::AggregateCodecBindingParams> decode(
      ConfigView view, const DecodeOptions& options, DecodeContext& context) {
    ObjectDecoder object{std::move(view), options, context};
    auto labels = object.required<std::vector<std::string>>("labels");
    if (!labels) {
      return tl::make_unexpected(std::move(labels.error()));
    }
    auto untouched = object.required<std::int32_t>("untouched");
    if (!untouched) {
      return tl::make_unexpected(std::move(untouched.error()));
    }
    auto finished = object.finish();
    if (!finished) {
      return tl::make_unexpected(std::move(finished.error()));
    }
    return holoscan::params::AggregateCodecBindingParams{.labels = std::move(*labels),
                                                         .untouched = {.value = *untouched}};
  }

  static ConfigResult<ConfigValue> encode(
      const holoscan::params::AggregateCodecBindingParams& value, const EncodeOptions& options,
      EncodeContext& context) {
    auto labels = context.encode_child<std::vector<std::string>>("labels", value.labels, options);
    if (!labels) {
      return tl::make_unexpected(std::move(labels.error()));
    }
    auto untouched =
        context.encode_child<std::int32_t>("untouched", value.untouched.value, options);
    if (!untouched) {
      return tl::make_unexpected(std::move(untouched.error()));
    }
    const std::array entries{ConfigObjectEntry{"labels", *labels},
                             ConfigObjectEntry{"untouched", *untouched}};
    return ConfigValue::try_object(entries);
  }
};

// Required schema-derived codec for nested inference settings.
// Representative test: RequiredNestedSchemaCodecNormalizesAliasDuringBinding.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::SchemaCodecInferenceParams>
    : holoscan::params::RequiredParamSchemaCodec<holoscan::params::SchemaCodecInferenceParams> {};

// Defaulted schema-derived codec for the outer inference parameter carrier.
// Representative test: ParamSchemaCodecsApplyFieldPresenceWithoutRepeatingFields.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::SchemaCodecParams>
    : holoscan::params::DefaultedParamSchemaCodec<holoscan::params::SchemaCodecParams> {};

// Defaulted schema-derived codec for structured canonical field paths.
// Representative test: SchemaDerivedCodecMaterializesStructuredCanonicalPaths.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::StructuredSchemaCodecParams>
    : holoscan::params::DefaultedParamSchemaCodec<holoscan::params::StructuredSchemaCodecParams> {};

// Defaulted schema-derived codec used to exercise nested unknown-field policy.
// Representative test: DefaultedNestedSchemaCodecNormalizesAliasUnderBindingPolicies.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::DefaultedSchemaCodecInferenceParams>
    : holoscan::params::DefaultedParamSchemaCodec<
          holoscan::params::DefaultedSchemaCodecInferenceParams> {};

// Required schema-derived codec for literal prefix-overlap paths.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForStrictPrefixOverlap.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::PrefixOverlapSchemaCodecParams>
    : holoscan::params::RequiredParamSchemaCodec<holoscan::params::PrefixOverlapSchemaCodecParams> {
};

// Required schema-derived codec for distinct keys with equal decoded spellings.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForDecodedEqualPaths.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::DecodedEqualSchemaCodecParams>
    : holoscan::params::RequiredParamSchemaCodec<holoscan::params::DecodedEqualSchemaCodecParams> {
};

// Defaulted schema-derived codec for alias/canonical prefix overlap.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForAliasCanonicalPrefixOverlap.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::AliasCanonicalPrefixSchemaCodecParams>
    : holoscan::params::DefaultedParamSchemaCodec<
          holoscan::params::AliasCanonicalPrefixSchemaCodecParams> {};

// Test codec that charges a large shared allocation budget per child.
// Representative test: StructuredSchemaCodecSharesBudgetAndRemapsChildErrorPaths.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::StructuredBudgetValue> {
  static constexpr std::size_t kAllocationCharge = 34U * 1024U * 1024U;

  static ConfigResult<holoscan::params::StructuredBudgetValue> decode(ConfigView view,
                                                                      const DecodeOptions& options,
                                                                      DecodeContext& context) {
    auto charged = context.charge_allocation(kAllocationCharge);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    auto decoded = ConfigCodec<std::int32_t>::decode(std::move(view), options, context);
    if (!decoded) {
      return tl::make_unexpected(decoded.error());
    }
    return holoscan::params::StructuredBudgetValue{.value = *decoded};
  }

  static ConfigResult<ConfigValue> encode(const holoscan::params::StructuredBudgetValue& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    auto charged = context.charge_allocation(kAllocationCharge);
    if (!charged) {
      return tl::make_unexpected(charged.error());
    }
    return ConfigCodec<std::int32_t>::encode(value.value, options, context);
  }
};

// Required schema-derived codec that composes two budget-charging children.
// Representative test: StructuredSchemaCodecSharesBudgetAndRemapsChildErrorPaths.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::StructuredBudgetSchemaCodecParams>
    : holoscan::params::RequiredParamSchemaCodec<
          holoscan::params::StructuredBudgetSchemaCodecParams> {};

// Handwritten codec whose field spelling deliberately differs from the parameter schema alias.
// Representative test: HandwrittenNestedCodecDoesNotNormalizeSchemaAliases.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::HandwrittenAliasParams>
    : holoscan::config::ObjectCodec<holoscan::params::HandwrittenAliasParams> {
  inline static constexpr auto fields = std::tuple{
      holoscan::config::object_field("engine", &holoscan::params::HandwrittenAliasParams::engine)};
};

namespace holoscan::params {
namespace {

static_assert(config::ConfigCodable<SchemaCodecInferenceParams>);
static_assert(config::ConfigCodable<SchemaCodecParams>);
static_assert(config::ConfigCodable<StructuredSchemaCodecParams>);
static_assert(config::ConfigCodable<DefaultedSchemaCodecInferenceParams>);
static_assert(config::ConfigCodable<PrefixOverlapSchemaCodecParams>);
static_assert(config::ConfigCodable<DecodedEqualSchemaCodecParams>);
static_assert(config::ConfigCodable<AliasCanonicalPrefixSchemaCodecParams>);
static_assert(config::ConfigCodable<StructuredBudgetSchemaCodecParams>);
static_assert(config::ConfigCodable<HandwrittenAliasParams>);

// Carries alias-prefix parameters without compute behavior for graph binding checks.
// Representative test: SchemaDerivedCodecUsesLiteralKeysForAliasCanonicalPrefixOverlap.
class AliasCanonicalPrefixBindingOperator final
    : public Operator<AliasCanonicalPrefixBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// REQ: PARAM-038
TEST(ParameterConfigBinding, ParamSchemaCodecsApplyFieldPresenceWithoutRepeatingFields) {
  // Key setup: Encode complete, defaulted, and missing-required schema values.
  const SchemaCodecParams source{.inference = {.iterations = 7, .engine = "optimized"},
                                 .enabled = false};
  auto encoded = config::try_encode_config(source);

  // Key assertion: Generated codecs honor canonical names, defaults, and required presence.
  ASSERT_TRUE(encoded);
  auto inference = encoded->view().try_member("inference");
  ASSERT_TRUE(inference);
  auto engine_name = inference->try_find_member("engine_name");
  ASSERT_TRUE(engine_name);
  EXPECT_TRUE(engine_name->has_value());
  auto member_name = inference->try_find_member("engine");
  ASSERT_TRUE(member_name);
  EXPECT_FALSE(member_name->has_value());
  auto decoded = config::try_decode_config<SchemaCodecParams>(encoded->view());
  ASSERT_TRUE(decoded);
  EXPECT_EQ(*decoded, source);

  const std::array outer_entries{config::ConfigObjectEntry{"inference", inference->to_value()}};
  auto without_enabled = config::ConfigValue::try_object(outer_entries);
  ASSERT_TRUE(without_enabled);
  auto defaults = config::try_decode_config<SchemaCodecParams>(without_enabled->view());
  ASSERT_TRUE(defaults);
  EXPECT_TRUE(defaults->enabled);

  auto iterations = config::ConfigValue::try_signed_integer(5);
  ASSERT_TRUE(iterations);
  const std::array inference_entries{config::ConfigObjectEntry{"iterations", *iterations}};
  auto partial_inference = config::ConfigValue::try_object(inference_entries);
  ASSERT_TRUE(partial_inference);
  auto missing_required =
      config::try_decode_config<SchemaCodecInferenceParams>(partial_inference->view());
  ASSERT_FALSE(missing_required);
  EXPECT_EQ(missing_required.error().path(), "/engine_name");
}

// Params schema whose required field carries a 0-100 range and legacy alias.
// Representative test: GraphAttachmentAppliesSparsePatchOverFinalBaseline.
struct ConfigBindingParams {
  std::int32_t required{1};
  std::int32_t optional{2};
};

// Duration-valued parameter carrier with millisecond field constraints.
// Representative test: GraphAttachmentValidatesDurationConstraintsInFieldUnits.
struct DurationBindingParams {
  std::chrono::milliseconds deadline{3};
};

HOLOSCAN_PARAM_SCHEMA(DurationBindingParams, param_schema("holoscan.test.DurationBindingParams", 1),
                      field<&DurationBindingParams::deadline>("deadline")
                          .range(std::chrono::milliseconds{1}, std::chrono::milliseconds{10}));

HOLOSCAN_PARAM_SCHEMA(ConfigBindingParams, param_schema("holoscan.test.ConfigBindingParams", 1),
                      field<&ConfigBindingParams::required>("required")
                          .runtime_mutable()
                          .external_tuning()
                          .range(0, 100)
                          .alias("legacy_required"),
                      field<&ConfigBindingParams::optional>("optional"));

// No-op operator carrying ConfigBindingParams. Used to register operators in binding test graphs
// without any real compute logic.
// Representative test: TypedLookupHandleRetainsCreationHandleProvenance.
class ConfigBindingOperator final : public Operator<ConfigBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Carries constrained duration parameters for config-to-field unit conversion checks.
// Representative test: GraphAttachmentValidatesDurationConstraintsInFieldUnits.
class DurationBindingOperator final : public Operator<DurationBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Carries nested schema-codec parameters for sparse overlay checks.
// Representative test: NestedRequiredParamSchemaCodecPreservesSparseOverlayBaseline.
class SchemaCodecOperator final : public Operator<SchemaCodecParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Carries structured-path schema parameters for graph attachment and compile checks.
// Representative test: SchemaDerivedCodecMaterializesStructuredCanonicalPaths.
class StructuredSchemaCodecOperator final : public Operator<StructuredSchemaCodecParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Mixed container and nested-codec fields used to verify field-local sparse replacement.
// Representative test: GraphAttachmentStagesOnlyTheConfiguredTypedField.
struct ContainerBindingParams {
  std::vector<std::string> labels{"baseline"};
  std::map<std::string, double> gains{{"retained", 1.0}};
  std::vector<ConfigBindingHistoryTerm> history{{.input_name = "baseline"}};
  SchemaCodecInferenceParams inference;
  std::int32_t untouched{7};
};

HOLOSCAN_PARAM_SCHEMA(
    ContainerBindingParams, param_schema("holoscan.test.ContainerBindingParams", 1),
    field<&ContainerBindingParams::labels>("labels"),
    field<&ContainerBindingParams::gains>("gains"),
    field<&ContainerBindingParams::history>("history"),
    field<&ContainerBindingParams::inference>("inference").nested<SchemaCodecInferenceParams>(),
    field<&ContainerBindingParams::untouched>("untouched"));

// Carries mixed container parameters without compute behavior for binding checks.
// Representative test: GraphAttachmentStagesOnlyTheConfiguredTypedField.
class ContainerBindingOperator final : public Operator<ContainerBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Wraps defaulted nested inference settings while retaining an unrelated baseline field.
// Representative test: DefaultedNestedSchemaCodecNormalizesAliasUnderBindingPolicies.
struct DefaultedNestedAliasBindingParams {
  DefaultedSchemaCodecInferenceParams inference;
  std::int32_t untouched{7};
};

HOLOSCAN_PARAM_SCHEMA(DefaultedNestedAliasBindingParams,
                      param_schema("holoscan.test.DefaultedNestedAliasBindingParams", 1),
                      field<&DefaultedNestedAliasBindingParams::inference>("inference")
                          .nested<DefaultedSchemaCodecInferenceParams>(),
                      field<&DefaultedNestedAliasBindingParams::untouched>("untouched"));

// Carries defaulted nested alias parameters for policy-sensitive graph bindings.
// Representative test: DefaultedNestedSchemaCodecNormalizesAliasUnderBindingPolicies.
class DefaultedNestedAliasBindingOperator final
    : public Operator<DefaultedNestedAliasBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Carries handwritten nested codec parameters for alias-boundary checks.
// Representative test: HandwrittenNestedCodecDoesNotNormalizeSchemaAliases.
class HandwrittenNestedAliasBindingOperator final
    : public Operator<HandwrittenNestedAliasBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};
template <typename HandleT, typename ValueT>
concept AcceptsSetParams =
    requires(HandleT handle, ValueT value) { handle.set_params(std::move(value)); };

template <typename BindingT>
concept AcceptsGraphInferredConfigAttachment =
    requires(const config::Document& source, BindingT binding) { set_params(source, {binding}); };

template <typename BindingT>
concept AcceptsExplicitGraphConfigAttachment =
    requires(Graph& graph, const config::Document& source, BindingT binding) {
      set_params(graph, source, {binding});
    };

using ConfigBindingHandle = OperatorHandle<ConfigBindingOperator>;
static_assert(
    std::same_as<decltype(std::declval<const PreparedParams<ConfigBindingParams>&>().operator->()),
                 const ConfigBindingParams*>);
static_assert(std::same_as<decltype(*std::declval<const PreparedParams<ConfigBindingParams>&>()),
                           const ConfigBindingParams&>);
static_assert(
    std::same_as<decltype(std::declval<const PreparedParams<ConfigBindingParams>&>().value()),
                 const ConfigBindingParams&>);
static_assert(std::same_as<decltype(std::declval<PreparedParams<ConfigBindingParams>&&>().take()),
                           ConfigBindingParams>);
static_assert(!std::convertible_to<PreparedParams<ConfigBindingParams>, ConfigBindingParams>);
static_assert(AcceptsSetParams<ConfigBindingHandle, PreparedParams<ConfigBindingParams>>);
static_assert(!AcceptsSetParams<ConfigBindingHandle, PreparedParams<ContainerBindingParams>>);
static_assert(!std::default_initializable<ParamConfigBinding>);
static_assert(
    !std::constructible_from<ParamConfigBinding, OperatorPath, config::Path, ParamBindingMode>);
static_assert(!AcceptsGraphInferredConfigAttachment<ParamConfigBinding>);
static_assert(AcceptsExplicitGraphConfigAttachment<ParamConfigBinding>);

// Carries an aggregate-only codec whose omitted field lacks a field-level route.
// Representative test: GraphAttachmentRejectsAggregateOnlyFieldCodec.
class AggregateCodecBindingOperator final : public Operator<AggregateCodecBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Contains a field with no portable field codec to exercise unavailable-route rejection.
// Representative test: TypedBindingReportsUnavailableFieldCodec.
struct UnsupportedBindingParams {
  struct Value {
    std::int32_t number{};
  } value;
};

HOLOSCAN_PARAM_SCHEMA(UnsupportedBindingParams,
                      param_schema("holoscan.test.UnsupportedBindingParams", 1),
                      field<&UnsupportedBindingParams::value>("value"));

// Carries UnsupportedBindingParams for graph attachment failure checks.
// Representative test: TypedBindingReportsUnavailableFieldCodec.
class UnsupportedBindingOperator final : public Operator<UnsupportedBindingParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Changes its input name from a parameter to expose declaration-dependent replacement.
// Representative test: CompileRejectsParameterDependentDeclarationChanges.
class DeclarationBindingOperator final : public Operator<DeclarationBindingParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, params().input_index == 0 ? "input" : "changed-input");
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::int32_t> input;
};

class CudaGraphEligibilityBindingOperator final : public Operator<DeclarationBindingParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.cuda_graph_capture_eligible(params().input_index == 0U);
  }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Test DSL helpers: thin assertion-wrapped constructors for config values, documents, and paths.
// value_or_null: unwraps a ConfigResult and asserts success; returns null on failure.
// object/array/document/path: build their respective types with inline assertion checks.
[[nodiscard]] config::ConfigValue value_or_null(config::ConfigResult<config::ConfigValue> result) {
  EXPECT_TRUE(result.has_value());
  return result ? std::move(*result) : config::ConfigValue::null();
}

[[nodiscard]] config::ConfigValue object(
    std::initializer_list<std::pair<std::string_view, config::ConfigValue>> values) {
  std::vector<std::string> keys;
  std::vector<config::ConfigValue> owned_values;
  keys.reserve(values.size());
  owned_values.reserve(values.size());
  for (const auto& [key, value] : values) {
    keys.emplace_back(key);
    owned_values.push_back(value);
  }
  std::vector<config::ConfigObjectEntry> entries;
  entries.reserve(values.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    entries.push_back(config::ConfigObjectEntry{.key = keys[index], .value = owned_values[index]});
  }
  return value_or_null(config::ConfigValue::try_object(entries));
}

[[nodiscard]] config::ConfigValue array(std::initializer_list<config::ConfigValue> values) {
  const std::vector<config::ConfigValue> owned{values};
  return value_or_null(config::ConfigValue::try_array(owned));
}

[[nodiscard]] config::Document document(config::ConfigValue root) {
  auto result = config::Document::try_from_value(std::move(root));
  EXPECT_TRUE(result.has_value());
  return result ? std::move(*result)
                : config::Document::try_from_value(config::ConfigValue::null()).value();
}

// Test DSL shorthand for paths whose literals are valid by construction.
[[nodiscard]] config::Path path(std::string_view text) {
  return config::Path::parse(text);
}

// REQ: PARAM-037
// REQ: PARAM-038
//   graph | rejected_graph | ignored_graph
//   `-- op(StructuredSchemaCodecOperator)
TEST(ParameterConfigBinding, SchemaDerivedCodecMaterializesStructuredCanonicalPaths) {
  // Key setup: Round-trip structured paths, then bind valid, rejected, and ignore-policy graphs.
  const StructuredSchemaCodecParams source{.iterations = 7, .fine_gain = 2.5, .enabled = false};

  auto encoded = config::try_encode_config(source);

  // Key assertion: Canonical paths, provenance, compiled values, and unknown-field policy agree.
  ASSERT_TRUE(encoded.has_value());
  auto tuning = encoded->view().try_member("tuning");
  ASSERT_TRUE(tuning.has_value());
  auto iterations = tuning->try_member("iterations");
  ASSERT_TRUE(iterations.has_value());
  auto decoded_iterations = iterations->try_as_signed_integer();
  ASSERT_TRUE(decoded_iterations.has_value());
  EXPECT_EQ(*decoded_iterations, 7);
  auto fine_gain = tuning->try_member("gain/fine");
  ASSERT_TRUE(fine_gain.has_value());
  auto decoded_gain = fine_gain->try_as_floating();
  ASSERT_TRUE(decoded_gain.has_value());
  EXPECT_DOUBLE_EQ(*decoded_gain, 2.5);
  auto encoded_literal = encoded->view().try_find_member("tuning/gain%2Ffine");
  ASSERT_TRUE(encoded_literal.has_value());
  EXPECT_FALSE(encoded_literal->has_value());
  auto decoded = config::try_decode_config<StructuredSchemaCodecParams>(encoded->view());
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(*decoded, source);

  Graph graph{"config-binding-structured-schema-codec"};
  auto op = graph.op<StructuredSchemaCodecOperator>("op");
  op.set_params(StructuredSchemaCodecParams{.iterations = 3, .fine_gain = 1.5, .enabled = false});
  const config::Document config_source = document(object({
      {"managed",
       object({{"tuning",
                object({{"iterations", value_or_null(config::ConfigValue::try_signed_integer(9))},
                        {"gain/fine", value_or_null(config::ConfigValue::try_floating(3.5))}})}})},
  }));

  auto rebound = set_params(graph, config_source, {bind(op, path("/managed"))});

  ASSERT_TRUE(rebound.has_value()) << rebound.error().token();
  ASSERT_EQ(rebound->config_fields.size(), 2U);
  const ParamSchemaDescriptor* const schema =
      param_schema_descriptor<StructuredSchemaCodecParams>();
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(rebound->config_fields[0].source_path, "/managed/tuning/iterations");
  EXPECT_EQ(rebound->config_fields[0].target_field_id, schema->fields[0].id);
  EXPECT_EQ(rebound->config_fields[0].target_full_id, schema->fields[0].full_id);
  EXPECT_EQ(rebound->config_fields[0].origin, ParamFieldRebindOrigin::kCaptured);
  EXPECT_EQ(rebound->config_fields[1].source_path, "/managed/tuning/gain~1fine");
  EXPECT_EQ(rebound->config_fields[1].target_field_id, schema->fields[1].id);
  EXPECT_EQ(rebound->config_fields[1].target_full_id, schema->fields[1].full_id);
  EXPECT_EQ(rebound->config_fields[1].origin, ParamFieldRebindOrigin::kCaptured);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* plan_state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(plan_state, nullptr);
  const auto* value = static_cast<const StructuredSchemaCodecParams*>(
      plan_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->iterations, 9);
  EXPECT_DOUBLE_EQ(value->fine_gain, 3.5);
  EXPECT_FALSE(value->enabled);

  const config::Document config_with_unknown = document(object({
      {"managed",
       object({{"tuning",
                object({{"iterations", value_or_null(config::ConfigValue::try_signed_integer(9))},
                        {"gain/fine", value_or_null(config::ConfigValue::try_floating(3.5))}})},
               {"unknown", value_or_null(config::ConfigValue::try_signed_integer(1))}})},
  }));
  Graph rejected_graph{"config-binding-structured-schema-codec-rejected"};
  auto rejected_op = rejected_graph.op<StructuredSchemaCodecOperator>("op");
  auto rejected =
      set_params(rejected_graph, config_with_unknown, {bind(rejected_op, path("/managed"))});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().token(), "PARAM_CONFIG_FIELD_UNKNOWN");

  Graph ignored_graph{"config-binding-structured-schema-codec-ignored"};
  auto ignored_op = ignored_graph.op<StructuredSchemaCodecOperator>("op");
  ignored_op.set_params(
      StructuredSchemaCodecParams{.iterations = 3, .fine_gain = 1.5, .enabled = false});
  ParamConfigDecodeOptions ignore_unknown;
  ignore_unknown.values = config::DecodeOptions{config::MissingValuePolicy::kReject,
                                                config::UnknownFieldPolicy::kIgnore};
  auto ignored = set_params(
      ignored_graph, config_with_unknown, {bind(ignored_op, path("/managed"))}, ignore_unknown);

  ASSERT_TRUE(ignored.has_value()) << ignored.error().token();
  const ExecutionPlan ignored_plan = compile(ignored_graph);
  ASSERT_TRUE(ignored_plan.ok()) << ignored_plan.json();
  const auto* ignored_state = holoscan::detail::ExecutionPlanAccess::state(ignored_plan);
  ASSERT_NE(ignored_state, nullptr);
  const auto* ignored_value = static_cast<const StructuredSchemaCodecParams*>(
      ignored_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(ignored_value, nullptr);
  EXPECT_EQ(*ignored_value, *value);
}

// REQ: PARAM-038
TEST(ParameterConfigBinding, SchemaDerivedCodecUsesLiteralKeysForStrictPrefixOverlap) {
  // Key setup: Encode sibling literal keys where one spelling is a strict prefix.
  const PrefixOverlapSchemaCodecParams source{.tuning = 7, .iterations = 9};

  auto encoded = config::try_encode_config(source);

  // Key assertion: Both literal keys survive and missing descendants report an escaped path.
  ASSERT_TRUE(encoded.has_value());
  auto tuning = encoded->view().try_member("tuning");
  ASSERT_TRUE(tuning.has_value());
  EXPECT_EQ(tuning->kind(), config::ConfigValueKind::kSignedInteger);
  auto iterations = encoded->view().try_member("tuning/iterations");
  ASSERT_TRUE(iterations.has_value());
  EXPECT_EQ(iterations->kind(), config::ConfigValueKind::kSignedInteger);
  auto decoded = config::try_decode_config<PrefixOverlapSchemaCodecParams>(encoded->view());
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(*decoded, source);

  const config::ConfigValue missing = object({
      {"tuning", value_or_null(config::ConfigValue::try_signed_integer(7))},
  });
  auto rejected = config::try_decode_config<PrefixOverlapSchemaCodecParams>(missing.view());
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().path(), "/tuning~1iterations");
}

// REQ: PARAM-038
TEST(ParameterConfigBinding, SchemaDerivedCodecUsesLiteralKeysForDecodedEqualPaths) {
  // Key setup: Encode distinct literal keys whose decoded spellings compare equal.
  const DecodedEqualSchemaCodecParams source{.plain = 7, .escaped = 9};

  auto encoded = config::try_encode_config(source);

  // Key assertion: Literal identity keeps both values distinct through round-trip decoding.
  ASSERT_TRUE(encoded.has_value());
  auto plain = encoded->view().try_member("foo");
  ASSERT_TRUE(plain.has_value());
  EXPECT_EQ(plain->kind(), config::ConfigValueKind::kSignedInteger);
  auto escaped = encoded->view().try_member("%66oo");
  ASSERT_TRUE(escaped.has_value());
  EXPECT_EQ(escaped->kind(), config::ConfigValueKind::kSignedInteger);
  auto decoded = config::try_decode_config<DecodedEqualSchemaCodecParams>(encoded->view());
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(*decoded, source);
}

// REQ: PARAM-038
//   graph
//   `-- op(AliasCanonicalPrefixBindingOperator)
TEST(ParameterConfigBinding, SchemaDerivedCodecUsesLiteralKeysForAliasCanonicalPrefixOverlap) {
  // Key setup: Round-trip alias-prefix fields and bind their nested wrapper into one Operator.
  const AliasCanonicalPrefixSchemaCodecParams source{.aliased = 7, .descendant = 9};

  auto encoded = config::try_encode_config(source);

  // Key assertion: Alias and descendant remain distinct with canonical field provenance.
  ASSERT_TRUE(encoded.has_value());
  auto aliased = encoded->view().try_member("alpha");
  ASSERT_TRUE(aliased.has_value());
  auto decoded_aliased = aliased->try_as_signed_integer();
  ASSERT_TRUE(decoded_aliased.has_value());
  EXPECT_EQ(*decoded_aliased, 7);
  auto descendant = encoded->view().try_member("foo/bar");
  ASSERT_TRUE(descendant.has_value());
  auto decoded_descendant = descendant->try_as_signed_integer();
  ASSERT_TRUE(decoded_descendant.has_value());
  EXPECT_EQ(*decoded_descendant, 9);
  auto alias = encoded->view().try_find_member("foo");
  ASSERT_TRUE(alias.has_value());
  EXPECT_FALSE(alias->has_value());
  auto decoded = config::try_decode_config<AliasCanonicalPrefixSchemaCodecParams>(encoded->view());
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(*decoded, source);

  Graph graph{"config-binding-alias-canonical-prefix"};
  auto op = graph.op<AliasCanonicalPrefixBindingOperator>("op");
  op.set_params(AliasCanonicalPrefixBindingParams{});
  const config::Document config_source = document(object({
      {"managed",
       object(
           {{"value",
             object({{"foo", value_or_null(config::ConfigValue::try_signed_integer(11))},
                     {"foo/bar", value_or_null(config::ConfigValue::try_signed_integer(13))}})}})},
  }));

  auto rebound = set_params(graph, config_source, {bind(op, path("/managed"))});

  ASSERT_TRUE(rebound.has_value()) << rebound.error().token();
  ASSERT_EQ(rebound->config_fields.size(), 1U);
  const ParamSchemaDescriptor* const schema =
      param_schema_descriptor<AliasCanonicalPrefixBindingParams>();
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(rebound->config_fields.front().source_path, "/managed/value");
  EXPECT_EQ(rebound->config_fields.front().target_field_id, schema->fields[0].id);
  EXPECT_EQ(rebound->config_fields.front().target_full_id, schema->fields[0].full_id);
  EXPECT_EQ(rebound->config_fields.front().origin, ParamFieldRebindOrigin::kCaptured);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* plan_state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(plan_state, nullptr);
  const auto* value = static_cast<const AliasCanonicalPrefixBindingParams*>(
      plan_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->value.aliased, 11);
  EXPECT_EQ(value->value.descendant, 13);
}

// REQ: PARAM-038
TEST(ParameterConfigBinding, StructuredSchemaCodecSharesBudgetAndRemapsChildErrorPaths) {
  // Key setup: Encode and decode two structured children against one deliberately small budget.
  const StructuredBudgetSchemaCodecParams source{.first = {.value = 7}, .second = {.value = 9}};

  auto encoded = config::try_encode_config(source);

  // Key assertion: Both directions exhaust the shared budget at the remapped second-child path.
  ASSERT_FALSE(encoded.has_value());
  EXPECT_EQ(encoded.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(encoded.error().path(), "/budget/second");

  const config::ConfigValue input = object({
      {"budget",
       object({{"first", value_or_null(config::ConfigValue::try_signed_integer(7))},
               {"second", value_or_null(config::ConfigValue::try_signed_integer(9))}})},
  });
  auto decoded = config::try_decode_config<StructuredBudgetSchemaCodecParams>(input.view());
  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().code(), config::ConfigErrorCode::kLimitExceeded);
  EXPECT_EQ(decoded.error().path(), "/budget/second");
}

// REQ: PARAM-037
// REQ: PARAM-038
//   graph
//   `-- op(ContainerBindingOperator)
TEST(ParameterConfigBinding, RequiredNestedSchemaCodecNormalizesAliasDuringBinding) {
  // Key setup: Attach an alias-spelled required nested object over a typed baseline.
  Graph graph{"config-binding-required-nested-alias"};
  auto op = graph.op<ContainerBindingOperator>("op");
  op.set_params(ContainerBindingParams{.inference = {.iterations = 3, .engine = "baseline"},
                                       .untouched = 11});
  const config::Document source = document(object({
      {"managed",
       object(
           {{"inference",
             object({{"iterations", value_or_null(config::ConfigValue::try_signed_integer(5))},
                     {"engine", value_or_null(config::ConfigValue::try_string("configured"))}})}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: Binding normalizes the alias and records the enclosing canonical field.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  ASSERT_EQ(attached->config_fields.size(), 1U);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* value = static_cast<const ContainerBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->inference,
            (SchemaCodecInferenceParams{.iterations = 5, .engine = "configured"}));
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<ContainerBindingParams>();
  ASSERT_NE(schema, nullptr);
  const auto field =
      std::ranges::find_if(schema->fields, [](const ParamFieldDescriptor& candidate) {
        return candidate.canonical_name == "inference";
      });
  ASSERT_NE(field, schema->fields.end());
  EXPECT_EQ(attached->config_fields.front().source_path, "/managed/inference");
  EXPECT_EQ(attached->config_fields.front().target_field_id, field->id);
  EXPECT_EQ(attached->config_fields.front().target_full_id, field->full_id);
  EXPECT_EQ(attached->config_fields.front().origin, ParamFieldRebindOrigin::kCaptured);
}

// REQ: PARAM-037
// REQ: PARAM-038
//   graph | ignored_graph
//   `-- op(DefaultedNestedAliasBindingOperator)
TEST(ParameterConfigBinding, DefaultedNestedSchemaCodecNormalizesAliasUnderBindingPolicies) {
  // Key setup: Bind alias-spelled defaulted nested values under reject and ignore policies.
  Graph graph{"config-binding-defaulted-nested-alias"};
  auto op = graph.op<DefaultedNestedAliasBindingOperator>("op");
  op.set_params(DefaultedNestedAliasBindingParams{
      .inference = {.engine = "baseline", .enabled = false}, .untouched = 11});
  const config::Document source = document(object({
      {"managed",
       object(
           {{"inference",
             object({{"engine", value_or_null(config::ConfigValue::try_string("configured"))}})}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: Both policies normalize aliases while preserving unrelated baseline fields.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  ASSERT_EQ(attached->config_fields.size(), 1U);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* value = static_cast<const DefaultedNestedAliasBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->inference.engine, "configured");
  EXPECT_TRUE(value->inference.enabled);
  EXPECT_EQ(value->untouched, 11);
  const ParamSchemaDescriptor* const schema =
      param_schema_descriptor<DefaultedNestedAliasBindingParams>();
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(attached->config_fields.front().source_path, "/managed/inference");
  EXPECT_EQ(attached->config_fields.front().target_field_id, schema->fields[0].id);
  EXPECT_EQ(attached->config_fields.front().target_full_id, schema->fields[0].full_id);
  EXPECT_EQ(attached->config_fields.front().origin, ParamFieldRebindOrigin::kCaptured);

  Graph ignored_graph{"config-binding-defaulted-nested-alias-ignore-policy"};
  auto ignored_op = ignored_graph.op<DefaultedNestedAliasBindingOperator>("op");
  ignored_op.set_params(DefaultedNestedAliasBindingParams{
      .inference = {.engine = "baseline", .enabled = false}, .untouched = 11});
  ParamConfigDecodeOptions ignore_unknown;
  ignore_unknown.values = config::DecodeOptions{config::MissingValuePolicy::kReject,
                                                config::UnknownFieldPolicy::kIgnore};
  auto ignored =
      set_params(ignored_graph, source, {bind(ignored_op, path("/managed"))}, ignore_unknown);

  ASSERT_TRUE(ignored.has_value()) << ignored.error().token();
  const ExecutionPlan ignored_plan = compile(ignored_graph);
  ASSERT_TRUE(ignored_plan.ok()) << ignored_plan.json();
  const auto* ignored_state = holoscan::detail::ExecutionPlanAccess::state(ignored_plan);
  ASSERT_NE(ignored_state, nullptr);
  const auto* ignored_value = static_cast<const DefaultedNestedAliasBindingParams*>(
      ignored_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(ignored_value, nullptr);
  EXPECT_EQ(ignored_value->inference, value->inference);
  EXPECT_EQ(ignored_value->untouched, value->untouched);
}

// REQ: PARAM-038
//   graph
//   `-- op(HandwrittenNestedAliasBindingOperator)
TEST(ParameterConfigBinding, HandwrittenNestedCodecDoesNotNormalizeSchemaAliases) {
  // Key setup: Bind a handwritten nested codec whose spelling happens to match a schema alias.
  Graph graph{"config-binding-handwritten-nested-alias"};
  auto op = graph.op<HandwrittenNestedAliasBindingOperator>("op");
  op.set_params(
      HandwrittenNestedAliasBindingParams{.handwritten = {.engine = "baseline"}, .untouched = 11});
  const config::Document source = document(object({
      {"managed",
       object(
           {{"handwritten",
             object({{"engine", value_or_null(config::ConfigValue::try_string("configured"))}})}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: The handwritten codec decodes literally and retains wrapper-field identity.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  ASSERT_EQ(attached->config_fields.size(), 1U);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* value = static_cast<const HandwrittenNestedAliasBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->handwritten.engine, "configured");
  EXPECT_EQ(value->untouched, 11);
  const ParamSchemaDescriptor* const schema =
      param_schema_descriptor<HandwrittenNestedAliasBindingParams>();
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(attached->config_fields.front().target_field_id, schema->fields[0].id);
  EXPECT_EQ(attached->config_fields.front().target_full_id, schema->fields[0].full_id);
}
// Composes one pre-bound ConfigBindingOperator for embedded transaction remapping checks.
// Representative test: EmbeddedGraphTransactionUsesRemappedTarget.
class AttachedConfigGraph final : public Graph {
 protected:
  void compose() override {
    auto op = this->op<ConfigBindingOperator>("op");
    op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
    const config::Document source = document(object({
        {"managed",
         object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
    }));
    auto attached = set_params(*this, source, {bind(op, path("/managed"))});
    if (!attached) {
      throw std::runtime_error(std::string(attached.error().token()));
    }
  }
};

[[nodiscard]] const ConfigBindingParams* compiled_config_binding_params(
    const holoscan::detail::ExecutionPlanState& state, std::string_view operator_name) {
  const auto vertex = std::ranges::find(state.graph.vertices(), operator_name, &VertexRecord::name);
  if (vertex == state.graph.vertices().end())
    return nullptr;
  const auto factory =
      std::ranges::find(state.executable_factories.operators,
                        vertex->id,
                        &holoscan::detail::ExecutableOperatorFactoryRecord::vertex);
  return factory == state.executable_factories.operators.end()
             ? nullptr
             : static_cast<const ConfigBindingParams*>(factory->factory->initial_parameter_value());
}

[[nodiscard]] const ContainerBindingParams* compiled_container_binding_params(
    const holoscan::detail::ExecutionPlanState& state, std::string_view operator_name) {
  const auto vertex = std::ranges::find(state.graph.vertices(), operator_name, &VertexRecord::name);
  if (vertex == state.graph.vertices().end())
    return nullptr;
  const auto factory =
      std::ranges::find(state.executable_factories.operators,
                        vertex->id,
                        &holoscan::detail::ExecutableOperatorFactoryRecord::vertex);
  return factory == state.executable_factories.operators.end()
             ? nullptr
             : static_cast<const ContainerBindingParams*>(
                   factory->factory->initial_parameter_value());
}

[[nodiscard]] std::shared_ptr<holoscan::detail::ParameterTransactionState>
copy_parameter_transaction(Graph& graph) {
  const auto snapshot = holoscan::detail::GraphCompilerAccess::canonical_snapshot(graph);
  auto factories = holoscan::detail::GraphCompilerAccess::executable_factories(graph, *snapshot);
  EXPECT_NE(factories.parameter_transaction, nullptr);
  return factories.parameter_transaction
             ? std::make_shared<holoscan::detail::ParameterTransactionState>(
                   *factories.parameter_transaction)
             : nullptr;
}

// REQ: PARAM-033
// REQ: PARAM-037
// REQ: PARAM-038
//   graph
//   `-- op(ContainerBindingOperator)
TEST(ParameterConfigBinding, NestedSchemaCodecRejectsCanonicalAliasDuplicateAtomically) {
  // Key setup: Supply canonical and alias spellings for the same required nested field.
  Graph graph{"config-binding-nested-alias-duplicate"};
  auto op = graph.op<ContainerBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object(
           {{"inference",
             object({{"iterations", value_or_null(config::ConfigValue::try_signed_integer(5))},
                     {"engine_name", value_or_null(config::ConfigValue::try_string("canonical"))},
                     {"engine", value_or_null(config::ConfigValue::try_string("alias"))}})}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: Duplicate contributions reject the whole attachment transaction.
  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().token(), "PARAM_CONFIG_FIELD_DUPLICATE");
}

// REQ: PARAM-038
TEST(ParameterConfigBinding, DirectNestedSchemaCodecDecodeRemainsCanonicalOnly) {
  // Key setup: Decode alias-only and canonical nested objects under strict and ignore policies.
  const config::ConfigValue alias_value = object({
      {"iterations", value_or_null(config::ConfigValue::try_signed_integer(5))},
      {"engine", value_or_null(config::ConfigValue::try_string("configured"))},
  });

  auto required_rejected =
      config::try_decode_config<SchemaCodecInferenceParams>(alias_value.view());

  // Key assertion: Direct codecs accept canonical names only and otherwise preserve defaults.
  ASSERT_FALSE(required_rejected.has_value());
  EXPECT_EQ(required_rejected.error().path(), "/engine_name");
  const config::DecodeOptions ignore_unknown{config::MissingValuePolicy::kReject,
                                             config::UnknownFieldPolicy::kIgnore};
  auto required_ignored =
      config::try_decode_config<SchemaCodecInferenceParams>(alias_value.view(), ignore_unknown);
  ASSERT_FALSE(required_ignored.has_value());
  EXPECT_EQ(required_ignored.error().path(), "/engine_name");

  const config::ConfigValue defaulted_alias = object({
      {"engine", value_or_null(config::ConfigValue::try_string("configured"))},
  });
  auto defaulted_ignored = config::try_decode_config<DefaultedSchemaCodecInferenceParams>(
      defaulted_alias.view(), ignore_unknown);
  ASSERT_TRUE(defaulted_ignored.has_value());
  EXPECT_EQ(defaulted_ignored->engine, "default");
  EXPECT_TRUE(defaulted_ignored->enabled);

  const config::ConfigValue canonical = object({
      {"engine_name", value_or_null(config::ConfigValue::try_string("configured"))},
  });
  auto canonical_decoded =
      config::try_decode_config<DefaultedSchemaCodecInferenceParams>(canonical.view());
  ASSERT_TRUE(canonical_decoded.has_value());
  EXPECT_EQ(canonical_decoded->engine, "configured");
}

// REQ: PARAM-033
// REQ: PARAM-038
//   graph
//   `-- op(SchemaCodecOperator)
TEST(ParameterConfigBinding, NestedRequiredParamSchemaCodecPreservesSparseOverlayBaseline) {
  // Key setup: Attach a complete required nested object over a baseline with an unrelated flag.
  Graph graph{"config-binding-schema-codec-overlay"};
  auto op = graph.op<SchemaCodecOperator>("op");
  op.set_params(
      SchemaCodecParams{.inference = {.iterations = 3, .engine = "baseline"}, .enabled = false});
  const config::Document source = document(object({
      {"managed",
       object({{"inference",
                object({{"iterations", value_or_null(config::ConfigValue::try_signed_integer(5))},
                        {"engine_name",
                         value_or_null(config::ConfigValue::try_string("configured"))}})}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: The nested object is replaced while the outer baseline flag is retained.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const auto* value = static_cast<const SchemaCodecParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->inference,
            (SchemaCodecInferenceParams{.iterations = 5, .engine = "configured"}));
  EXPECT_FALSE(value->enabled);
}

// REQ: PARAM-033
// REQ: PARAM-045
// REQ: COMPILE-073
// REQ: PARAM-048
//   graph
//   `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, GraphAttachmentAppliesSparsePatchOverFinalBaseline) {
  // Key setup: Apply the same sparse attachment before and after the final typed baseline.
  Graph graph{"config-binding-attachment"};
  auto op = graph.op<ConfigBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: Attachment order is irrelevant and only the selected field is replaced.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  ASSERT_EQ(attached->config_fields.size(), 1U);
  EXPECT_EQ(attached->config_fields.front().target_operator_path, "op");
  op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const auto* compiled = static_cast<const ConfigBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->required, 42);
  EXPECT_EQ(compiled->optional, 9);

  Graph baseline_first_graph{"config-binding-attachment"};
  auto baseline_first_op = baseline_first_graph.op<ConfigBindingOperator>("op");
  baseline_first_op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  auto baseline_first_attached =
      set_params(baseline_first_graph, source, {bind(baseline_first_op, path("/managed"))});
  ASSERT_TRUE(baseline_first_attached.has_value()) << baseline_first_attached.error().token();
  const ExecutionPlan baseline_first_plan = compile(baseline_first_graph);
  ASSERT_TRUE(baseline_first_plan.ok()) << baseline_first_plan.json();
  const auto* baseline_first_state =
      holoscan::detail::ExecutionPlanAccess::state(baseline_first_plan);
  ASSERT_NE(baseline_first_state, nullptr);
  const auto* baseline_first_compiled = static_cast<const ConfigBindingParams*>(
      baseline_first_state->executable_factories.operators.front()
          .factory->initial_parameter_value());
  ASSERT_NE(baseline_first_compiled, nullptr);
  EXPECT_EQ(baseline_first_compiled->required, compiled->required);
  EXPECT_EQ(baseline_first_compiled->optional, compiled->optional);
}

// REQ: PARAM-045
//   graph
//   `-- op(DurationBindingOperator)
TEST(ParameterConfigBinding, GraphAttachmentValidatesDurationConstraintsInFieldUnits) {
  // Key setup: Bind a nanosecond config scalar to a millisecond-constrained duration field.
  Graph graph{"config-binding-duration"};
  auto op = graph.op<DurationBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"deadline", value_or_null(config::ConfigValue::try_signed_integer(5000000))}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: The in-range value compiles as the expected duration unit.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* compiled = static_cast<const DurationBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->deadline, std::chrono::milliseconds{5});
}

// REQ: PARAM-045
//   graph
//   `-- op(DurationBindingOperator)
TEST(ParameterConfigBinding, CompileRejectsDurationOutsideFieldRange) {
  // Key setup: Attach a duration value that exceeds the field's millisecond range.
  Graph graph{"config-binding-invalid-duration"};
  auto op = graph.op<DurationBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"deadline", value_or_null(config::ConfigValue::try_signed_integer(50000000))}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  const ExecutionPlan plan = compile(graph);

  // Key assertion: Compile rejects the staged value with the stable override diagnostic.
  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_OVERRIDE_VALUE_INVALID"), std::string::npos) << plan.json();
}

// REQ: COMPILE-073
//   graph
//   `-- op(DeclarationBindingOperator) -- input | changed-input
TEST(ParameterConfigBinding, CompileRejectsParameterDependentDeclarationChanges) {
  // Key setup: Bind a parameter that changes the Operator's declared input name.
  Graph graph{"config-binding-declaration-change"};
  auto op = graph.op<DeclarationBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"input_index", value_or_null(config::ConfigValue::try_signed_integer(1))}})},
  }));
  auto attached = set_params(graph, source, {bind(op, path("/managed"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  const ExecutionPlan plan = compile(graph);

  // Key assertion: Compile rejects parameter-dependent structural declarations.
  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_OVERRIDE_DECLARATION_MISMATCH"), std::string::npos)
      << plan.json();
}

// REQ: OPERATOR-051
TEST(ParameterConfigBinding, CompileRejectsCudaGraphEligibilityFactoryReplayMismatch) {
  Graph graph{"config-binding-cuda-graph-eligibility-change"};
  auto op = graph.op<CudaGraphEligibilityBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"input_index", value_or_null(config::ConfigValue::try_signed_integer(1))}})},
  }));
  auto attached = set_params(graph, source, {bind(op, path("/managed"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  const ExecutionPlan plan = compile(graph);

  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_OVERRIDE_DECLARATION_MISMATCH"), std::string::npos)
      << plan.json();
}

// REQ: GRAPH-035
// REQ: PARAM-048
//   creation_graph                 lookup_graph
//   `-- op(ConfigBindingOperator)  `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, TypedLookupHandleRetainsCreationHandleProvenance) {
  // Key setup: Bind identical config through creation and name-resolved handles in peer graphs.
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));
  Graph creation_graph{"config-binding-provenance-parity"};
  auto creation_handle = creation_graph.op<ConfigBindingOperator>("op");
  auto creation = set_params(creation_graph, source, {bind(creation_handle, path("/managed"))});
  ASSERT_TRUE(creation.has_value()) << creation.error().token();

  Graph lookup_graph{"config-binding-provenance-parity"};
  (void)lookup_graph.op<ConfigBindingOperator>("op");
  auto lookup_handle = lookup_graph.handle<ConfigBindingOperator>("op");
  auto lookup = set_params(lookup_graph, source, {bind(lookup_handle, path("/managed"))});
  ASSERT_TRUE(lookup.has_value()) << lookup.error().token();

  // Key assertion: Both handle origins produce identical persisted field provenance.
  ASSERT_EQ(creation->config_fields.size(), 1U);
  ASSERT_EQ(lookup->config_fields.size(), 1U);
  const ParamConfigFieldProvenance& expected = creation->config_fields.front();
  const ParamConfigFieldProvenance& actual = lookup->config_fields.front();
  EXPECT_EQ(actual.document.semantic_version, expected.document.semantic_version);
  EXPECT_EQ(actual.document.semantic_digest, expected.document.semantic_digest);
  EXPECT_EQ(actual.document.has_raw_source, expected.document.has_raw_source);
  EXPECT_EQ(actual.document.raw_algorithm_version, expected.document.raw_algorithm_version);
  EXPECT_EQ(actual.document.raw_digest, expected.document.raw_digest);
  EXPECT_EQ(actual.document.raw_byte_size, expected.document.raw_byte_size);
  EXPECT_EQ(actual.document.has_base_raw_source, expected.document.has_base_raw_source);
  EXPECT_EQ(actual.document.base_raw_algorithm_version,
            expected.document.base_raw_algorithm_version);
  EXPECT_EQ(actual.document.base_raw_digest, expected.document.base_raw_digest);
  EXPECT_EQ(actual.document.base_raw_byte_size, expected.document.base_raw_byte_size);
  EXPECT_EQ(actual.source_path, expected.source_path);
  EXPECT_EQ(actual.source_ordinal, expected.source_ordinal);
  EXPECT_EQ(actual.target_operator_path, expected.target_operator_path);
  EXPECT_EQ(actual.target_schema_id, expected.target_schema_id);
  EXPECT_EQ(actual.target_schema_version, expected.target_schema_version);
  EXPECT_EQ(actual.target_field_id, expected.target_field_id);
  EXPECT_EQ(actual.target_full_id, expected.target_full_id);
  EXPECT_EQ(actual.origin, expected.origin);
}

// REQ: PARAM-048
//   source_graph                 destination_graph
//   `-- op(ConfigBindingOperator)  `-- (empty)
TEST(ParameterConfigBinding, GraphAttachmentRejectsBindingFromAnotherGraph) {
  // Key setup: Apply a binding captured from source_graph to destination_graph.
  Graph source_graph{"config-binding-source"};
  auto source_op = source_graph.op<ConfigBindingOperator>("op");
  Graph destination_graph{"config-binding-destination"};
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));

  auto attached = set_params(destination_graph, source, {bind(source_op, path("/managed"))});

  // Key assertion: Cross-Graph binding provenance is rejected.
  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().token(), "PARAM_CONFIG_GRAPH_MISMATCH");
}

// REQ: PARAM-048
//   original graph                replacement graph at reused address
//   `-- op(ConfigBindingOperator) `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, GraphAttachmentRejectsBindingAfterGraphAddressReuse) {
  // Key setup: Retain a binding across Graph destruction and reconstruction at one address.
  std::optional<Graph> graph{std::in_place, "config-binding-original"};
  auto original_op = graph->op<ConfigBindingOperator>("op");
  auto stale_binding = bind(original_op, path("/managed"));
  graph.reset();
  graph.emplace("config-binding-replacement");
  (void)graph->op<ConfigBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));

  auto attached = set_params(*graph, source, {stale_binding});

  // Key assertion: Stable Graph identity rejects the stale binding despite address reuse.
  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().token(), "PARAM_CONFIG_GRAPH_MISMATCH");
}

// REQ: PARAM-045
//   graph
//   |-- a_unrelated(ConfigBindingOperator)
//   `-- z_target(ConfigBindingOperator)
TEST(ParameterConfigBinding, GraphAttachmentSurvivesCanonicalVertexReordering) {
  // Key setup: Attach to z_target before adding an unrelated lexically earlier Operator.
  Graph graph{"config-binding-canonical-order"};
  auto target = graph.op<ConfigBindingOperator>("z_target");
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));
  auto attached = set_params(graph, source, {bind(target, path("/managed"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  auto unrelated = graph.op<ConfigBindingOperator>("a_unrelated");
  unrelated.set_params(ConfigBindingParams{.required = 11, .optional = 12});
  target.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const ExecutionPlan plan = compile(graph);

  // Key assertion: Canonical reordering preserves the configured target and unrelated baseline.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* compiled_target = compiled_config_binding_params(*state, "z_target");
  const auto* compiled_unrelated = compiled_config_binding_params(*state, "a_unrelated");
  ASSERT_NE(compiled_target, nullptr);
  ASSERT_NE(compiled_unrelated, nullptr);
  EXPECT_EQ(compiled_target->required, 42);
  EXPECT_EQ(compiled_target->optional, 9);
  EXPECT_EQ(compiled_unrelated->required, 11);
  EXPECT_EQ(compiled_unrelated->optional, 12);
}

// REQ: PARAM-045
//   graph
//   `-- op(ContainerBindingOperator)
TEST(ParameterConfigBinding, GraphAttachmentStagesOnlyTheConfiguredTypedField) {
  // Key setup: Attach only labels, then replace the final aggregate baseline before compile.
  Graph graph{"config-binding-typed-attachment"};
  auto op = graph.op<ContainerBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"labels",
                array({value_or_null(config::ConfigValue::try_string("left")),
                       value_or_null(config::ConfigValue::try_string("right"))})}})},
  }));
  auto attached = set_params(graph, source, {bind(op, path("/managed"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  op.set_params(ContainerBindingParams{
      .labels = {"final-baseline"}, .gains = {{"retained", 3.0}}, .untouched = 11});

  const ExecutionPlan plan = compile(graph);

  // Key assertion: Labels come from config while all other fields come from the final baseline.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* compiled = static_cast<const ContainerBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->labels, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(compiled->gains, (std::map<std::string, double>{{"retained", 3.0}}));
  EXPECT_EQ(compiled->untouched, 11);

  const ExecutionPlan repeated_plan = compile(graph);
  ASSERT_TRUE(repeated_plan.ok()) << repeated_plan.json();
  const auto* repeated_state = holoscan::detail::ExecutionPlanAccess::state(repeated_plan);
  ASSERT_NE(repeated_state, nullptr);
  const auto* repeated = static_cast<const ContainerBindingParams*>(
      repeated_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(repeated, nullptr);
  EXPECT_EQ(repeated->labels, compiled->labels);
}

// REQ: PARAM-045
// REQ: PARAM-048
//   graph
//   `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, SecondGraphTransactionRejectsWithoutReplacingFirst) {
  // Key setup: Attach one transaction, then attempt a second transaction for the same Operator.
  Graph graph{"config-binding-second-transaction"};
  auto op = graph.op<ConfigBindingOperator>("op");
  op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document first = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));
  const config::Document second = document(object({
      {"managed",
       object({{"optional", value_or_null(config::ConfigValue::try_signed_integer(88))}})},
  }));
  const std::array bindings{bind(op, path("/managed"))};
  auto attached = set_params(graph, first, bindings);
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  auto rejected = set_params(graph, second, bindings);

  // Key assertion: The second attachment rejects and the first remains authoritative.
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().token(), "PARAM_CONFIG_ATTACH_FAILED");
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const auto* compiled = static_cast<const ConfigBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->required, 42);
  EXPECT_EQ(compiled->optional, 9);
}

// REQ: PARAM-045
//   graph
//   `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, DuplicateTransactionTargetRejectsWithoutAttaching) {
  // Key setup: Bind two source paths to one Operator in a single transaction.
  Graph graph{"config-binding-duplicate-target"};
  auto op = graph.op<ConfigBindingOperator>("op");
  op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
      {"second", object({{"optional", value_or_null(config::ConfigValue::try_signed_integer(8))}})},
  }));

  auto rejected = set_params(graph, source, {bind(op, path("/first")), bind(op, path("/second"))});

  // Key assertion: Duplicate targets reject atomically and a later valid attachment succeeds.
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().token(), "PARAM_CONFIG_OPERATOR_DUPLICATE");
  const auto& config_error = rejected.error().config_cause();
  ASSERT_TRUE(config_error.has_value());
  EXPECT_EQ(config_error->path(), "/second");
  auto attached = set_params(graph, source, {bind(op, path("/first"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* compiled = static_cast<const ConfigBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->required, 41);
}

// REQ: PARAM-045
// REQ: COMPILE-073
// REQ: PARAM-048
//   graph
//   |-- first(ConfigBindingOperator)
//   `-- second(ConfigBindingOperator)
TEST(ParameterConfigBinding, MultiOperatorAttachmentIsAtomicAndKeepsGlobalFieldOrder) {
  // Key setup: Reject an invalid two-Operator attachment, then attach one valid field per target.
  Graph graph{"config-binding-multi-operator"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ConfigBindingOperator>("second");
  first.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  second.set_params(ConfigBindingParams{.required = 6, .optional = 8});
  const config::Document invalid_source = document(object({
      {"alpha", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
      {"beta", object({{"unknown", value_or_null(config::ConfigValue::try_signed_integer(1))}})},
  }));

  auto rejected =
      set_params(graph, invalid_source, {bind(first, path("/alpha")), bind(second, path("/beta"))});

  // Key assertion: Rejection is atomic and successful provenance follows global target order.
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().token(), "PARAM_CONFIG_FIELD_UNKNOWN");
  const auto unknown_error = rejected.error().config_cause();
  ASSERT_TRUE(unknown_error.has_value());
  EXPECT_EQ(unknown_error->path(), "/beta/unknown");
  EXPECT_NE(unknown_error->message().find("second"), std::string_view::npos);
  EXPECT_EQ(unknown_error->cause_count(), 0U);
  const std::string formatted = format_error(rejected.error());
  EXPECT_NE(formatted.find("/beta/unknown"), std::string::npos);
  EXPECT_NE(formatted.find("second"), std::string::npos);

  const config::Document invalid_type = document(object({
      {"alpha", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
      {"beta", object({{"required", value_or_null(config::ConfigValue::try_string("invalid"))}})},
  }));
  rejected =
      set_params(graph, invalid_type, {bind(first, path("/alpha")), bind(second, path("/beta"))});
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().token(), "PARAM_CONFIG_SCALAR_INVALID");
  const auto& type_error = rejected.error().config_cause();
  ASSERT_TRUE(type_error.has_value());
  ASSERT_EQ(type_error->cause_count(), 1U);
  ASSERT_NE(type_error->cause(0), nullptr);
  EXPECT_EQ(type_error->cause(0)->code(), config::ConfigErrorCode::kTypeMismatch);
  const config::Document valid_source = document(object({
      {"alpha", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
      {"beta", object({{"optional", value_or_null(config::ConfigValue::try_signed_integer(18))}})},
  }));
  auto attached =
      set_params(graph, valid_source, {bind(first, path("/alpha")), bind(second, path("/beta"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  ASSERT_EQ(attached->config_fields.size(), 2U);
  EXPECT_EQ(attached->config_fields[0].target_operator_path, "first");
  EXPECT_EQ(attached->config_fields[1].target_operator_path, "second");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 2U);
  const auto* first_params = static_cast<const ConfigBindingParams*>(
      state->executable_factories.operators[0].factory->initial_parameter_value());
  const auto* second_params = static_cast<const ConfigBindingParams*>(
      state->executable_factories.operators[1].factory->initial_parameter_value());
  ASSERT_NE(first_params, nullptr);
  ASSERT_NE(second_params, nullptr);
  EXPECT_EQ(first_params->required, 41);
  EXPECT_EQ(first_params->optional, 9);
  EXPECT_EQ(second_params->required, 6);
  EXPECT_EQ(second_params->optional, 18);
}

// REQ: PARAM-048
//   donor                       graph
//   |-- first(ConfigBindingOperator)  |-- first(ConfigBindingOperator)
//   `-- second(ConfigBindingOperator) `-- second(ConfigBindingOperator)
TEST(ParameterConfigBinding, CompileRejectsProvenanceFromAnotherTransactionTarget) {
  // Key setup: Swap donor provenance indices before attaching the synthetic transaction.
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
      {"second",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));
  Graph donor{"config-binding-provenance-donor"};
  auto donor_first = donor.op<ConfigBindingOperator>("first");
  auto donor_second = donor.op<ConfigBindingOperator>("second");
  auto report = set_params(
      donor, source, {bind(donor_first, path("/first")), bind(donor_second, path("/second"))});
  ASSERT_TRUE(report.has_value()) << report.error().token();

  Graph graph{"config-binding-provenance-target"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ConfigBindingOperator>("second");
  auto first_vertex = holoscan::detail::ParameterConfigBindingAccess::vertex(first);
  auto second_vertex = holoscan::detail::ParameterConfigBindingAccess::vertex(second);
  ASSERT_TRUE(first_vertex.has_value());
  ASSERT_TRUE(second_vertex.has_value());
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<ConfigBindingParams>();
  const ParamTypeDescriptor& type = param_type_descriptor<ConfigBindingParams>();
  const ParamFieldDescriptor& field = schema->fields.front();
  auto transaction = std::make_shared<holoscan::detail::ParameterTransactionState>();
  transaction->source = holoscan::detail::ParameterTransactionSource::kConfig;
  transaction->provenance.mode = ParamRebindMode::kSparseContributions;
  transaction->provenance.config_fields = report->config_fields;
  for (auto& provenance : transaction->provenance.config_fields) {
    provenance.target_operator_path.clear();
  }
  const auto staged_operator = [&](VertexId target,
                                   std::int32_t value,
                                   std::uint32_t provenance_index) {
    const ParamScalarValue scalar = ParamScalarValue::from(value);
    return holoscan::detail::ParameterTransactionOperatorState{
        .target = target,
        .local_type_token = type.local_type_token,
        .schema_id = std::string{schema->stable_id},
        .schema_version = schema->version,
        .structural_hash = schema->structural_hash,
        .contract_hash = schema->contract_hash,
        .fields = {holoscan::detail::ParameterTransactionFieldState{
            .field = field.id,
            .full_id = field.full_id,
            .origin = ParamFieldRebindOrigin::kCaptured,
            .config_provenance_index = provenance_index,
            .scalar_value = scalar,
            .staged_value_digest = holoscan::detail::parameter_transaction_scalar_digest(scalar),
        }},
        .provenance_begin = static_cast<std::uint32_t>(target == *first_vertex ? 0U : 1U),
        .provenance_count = 1U,
    };
  };
  transaction->operators.push_back(staged_operator(*first_vertex, 41, 1U));
  transaction->operators.push_back(staged_operator(*second_vertex, 42, 0U));
  auto attached = holoscan::detail::ParameterInputAccess::attach_transaction(
      graph, std::shared_ptr<const holoscan::detail::ParameterTransactionState>{transaction});
  ASSERT_TRUE(attached.has_value()) << attached.error().message;

  const ExecutionPlan plan = compile(graph);

  // Key assertion: Compile rejects provenance assigned to the wrong transaction target.
  EXPECT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_OVERRIDE_PROVENANCE_MISMATCH"), std::string::npos)
      << plan.json();
}

// REQ: PARAM-048
// REQ: COMPILE-073
//   donor                         graph
//   |-- first(ConfigBindingOperator)  |-- first(ConfigBindingOperator)
//   `-- second(ConfigBindingOperator) `-- second(ConfigBindingOperator)
TEST(ParameterConfigBinding, CompileRejectsCorruptedScalarDigestWithoutPartialReplacement) {
  // Key setup: Corrupt one staged scalar digest before attaching the copied transaction.
  Graph donor{"config-binding-scalar-digest-donor"};
  auto donor_first = donor.op<ConfigBindingOperator>("first");
  auto donor_second = donor.op<ConfigBindingOperator>("second");
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
      {"second",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));
  auto staged = set_params(
      donor, source, {bind(donor_first, path("/first")), bind(donor_second, path("/second"))});
  ASSERT_TRUE(staged.has_value()) << staged.error().token();
  auto transaction = copy_parameter_transaction(donor);
  ASSERT_NE(transaction, nullptr);
  const auto corrupted =
      std::ranges::find(transaction->operators,
                        1U,
                        &holoscan::detail::ParameterTransactionOperatorState::provenance_begin);
  ASSERT_NE(corrupted, transaction->operators.end());
  ASSERT_EQ(corrupted->fields.size(), 1U);
  corrupted->fields.front().staged_value_digest.value.front() ^= std::byte{1U};

  Graph graph{"config-binding-scalar-digest-target"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ConfigBindingOperator>("second");
  first.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  second.set_params(ConfigBindingParams{.required = 6, .optional = 8});
  auto attached = holoscan::detail::ParameterInputAccess::attach_transaction(
      graph,
      std::shared_ptr<const holoscan::detail::ParameterTransactionState>{std::move(transaction)});
  ASSERT_TRUE(attached.has_value()) << attached.error().message;

  const ExecutionPlan plan = compile(graph);

  // Key assertion: Compile rejects the digest and preserves both Operator baselines.
  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_OVERRIDE_VALUE_INVALID"), std::string::npos) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* compiled_first = compiled_config_binding_params(*state, "first");
  const auto* compiled_second = compiled_config_binding_params(*state, "second");
  ASSERT_NE(compiled_first, nullptr);
  ASSERT_NE(compiled_second, nullptr);
  EXPECT_EQ(compiled_first->required, 7);
  EXPECT_EQ(compiled_first->optional, 9);
  EXPECT_EQ(compiled_second->required, 6);
  EXPECT_EQ(compiled_second->optional, 8);
}

// REQ: PARAM-048
// REQ: COMPILE-073
//   donor                         graph
//   |-- first(ConfigBindingOperator)  |-- first(ConfigBindingOperator)
//   `-- second(ContainerBindingOperator) `-- second(ContainerBindingOperator)
TEST(ParameterConfigBinding, CompileRejectsCorruptedNonScalarDigestWithoutPartialReplacement) {
  // Key setup: Corrupt one staged container digest before attaching the copied transaction.
  Graph donor{"config-binding-non-scalar-digest-donor"};
  auto donor_first = donor.op<ConfigBindingOperator>("first");
  auto donor_second = donor.op<ContainerBindingOperator>("second");
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
      {"second",
       object({{"labels", array({value_or_null(config::ConfigValue::try_string("configured"))})}})},
  }));
  auto staged = set_params(
      donor, source, {bind(donor_first, path("/first")), bind(donor_second, path("/second"))});
  ASSERT_TRUE(staged.has_value()) << staged.error().token();
  auto transaction = copy_parameter_transaction(donor);
  ASSERT_NE(transaction, nullptr);
  const auto corrupted =
      std::ranges::find(transaction->operators,
                        1U,
                        &holoscan::detail::ParameterTransactionOperatorState::provenance_begin);
  ASSERT_NE(corrupted, transaction->operators.end());
  ASSERT_EQ(corrupted->fields.size(), 1U);
  ASSERT_TRUE(corrupted->fields.front().local_value);
  corrupted->fields.front().staged_value_digest.value.front() ^= std::byte{1U};

  Graph graph{"config-binding-non-scalar-digest-target"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ContainerBindingOperator>("second");
  first.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  second.set_params(ContainerBindingParams{
      .labels = {"baseline"}, .gains = {{"retained", 3.0}}, .untouched = 11});
  auto attached = holoscan::detail::ParameterInputAccess::attach_transaction(
      graph,
      std::shared_ptr<const holoscan::detail::ParameterTransactionState>{std::move(transaction)});
  ASSERT_TRUE(attached.has_value()) << attached.error().message;

  const ExecutionPlan plan = compile(graph);

  // Key assertion: Compile rejects the digest and preserves both Operator baselines.
  ASSERT_FALSE(plan.ok());
  EXPECT_NE(plan.json().find("PARAMETER_OVERRIDE_VALUE_INVALID"), std::string::npos) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto* compiled_first = compiled_config_binding_params(*state, "first");
  const auto* compiled_second = compiled_container_binding_params(*state, "second");
  ASSERT_NE(compiled_first, nullptr);
  ASSERT_NE(compiled_second, nullptr);
  EXPECT_EQ(compiled_first->required, 7);
  EXPECT_EQ(compiled_first->optional, 9);
  EXPECT_EQ(compiled_second->labels, (std::vector<std::string>{"baseline"}));
  EXPECT_EQ(compiled_second->gains, (std::map<std::string, double>{{"retained", 3.0}}));
  EXPECT_EQ(compiled_second->untouched, 11);
}

// REQ: PARAM-045
// REQ: PARAM-048
// REQ: COMPILE-073
//   canonical_graph               alias_graph
//   `-- op(ConfigBindingOperator) `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, TransactionSourceSpellingDoesNotChangeSemanticPlanIdentity) {
  // Key setup: Build equivalent transactions using a canonical field and its legacy alias.
  Graph canonical_graph{"equivalent-config-transaction"};
  auto canonical_op = canonical_graph.op<ConfigBindingOperator>("op");
  canonical_op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document canonical_source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));
  auto canonical_attached =
      set_params(canonical_graph, canonical_source, {bind(canonical_op, path("/managed"))});
  ASSERT_TRUE(canonical_attached.has_value()) << canonical_attached.error().token();
  const ExecutionPlan canonical_plan = compile(canonical_graph);
  ASSERT_TRUE(canonical_plan.ok()) << canonical_plan.json();

  Graph alias_graph{"equivalent-config-transaction"};
  auto alias_op = alias_graph.op<ConfigBindingOperator>("op");
  alias_op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document alias_source = document(object({
      {"managed",
       object({{"legacy_required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));
  auto alias_attached = set_params(alias_graph, alias_source, {bind(alias_op, path("/managed"))});
  ASSERT_TRUE(alias_attached.has_value()) << alias_attached.error().token();
  const ExecutionPlan alias_plan = compile(alias_graph);
  ASSERT_TRUE(alias_plan.ok()) << alias_plan.json();

  const auto* canonical_state = holoscan::detail::ExecutionPlanAccess::state(canonical_plan);
  const auto* alias_state = holoscan::detail::ExecutionPlanAccess::state(alias_plan);
  ASSERT_NE(canonical_state, nullptr);
  ASSERT_NE(alias_state, nullptr);
  const auto* canonical_params = static_cast<const ConfigBindingParams*>(
      canonical_state->executable_factories.operators.front().factory->initial_parameter_value());
  const auto* alias_params = static_cast<const ConfigBindingParams*>(
      alias_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(canonical_params, nullptr);
  ASSERT_NE(alias_params, nullptr);

  // Key assertion: Both spellings materialize the same value and semantic plan identity.
  EXPECT_EQ(canonical_params->required, 42);
  EXPECT_EQ(alias_params->required, 42);
  EXPECT_EQ(canonical_state->plan_generation, alias_state->plan_generation);
}

// REQ: PARAM-045
// REQ: COMPILE-073
// REQ: PARAM-048
//   parent
//   |-- a_parent(ConfigBindingOperator)
//   `-- child(AttachedConfigGraph)
//       `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, EmbeddedGraphTransactionUsesRemappedTarget) {
  // Key setup: Embed a Graph with an attached transaction beside an unrelated parent Operator.
  Graph parent{"config-binding-embedded-transaction"};
  auto child = parent.embed<AttachedConfigGraph>("child");
  ASSERT_TRUE(child);
  auto unrelated = parent.op<ConfigBindingOperator>("a_parent");
  unrelated.set_params(ConfigBindingParams{.required = 11, .optional = 12});

  const ExecutionPlan plan = compile(parent);

  // Key assertion: The child transaction targets child/op without changing the parent Operator.
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 2U);
  const auto* compiled_child = compiled_config_binding_params(*state, "op");
  const auto* compiled_unrelated = compiled_config_binding_params(*state, "a_parent");
  ASSERT_NE(compiled_child, nullptr);
  ASSERT_NE(compiled_unrelated, nullptr);
  EXPECT_EQ(compiled_child->required, 42);
  EXPECT_EQ(compiled_child->optional, 9);
  EXPECT_EQ(compiled_unrelated->required, 11);
  EXPECT_EQ(compiled_unrelated->optional, 12);
  const auto configured =
      std::ranges::find_if(state->parameter_generation_zero_fields,
                           [](const auto& field) { return field.config_source.has_value(); });
  ASSERT_NE(configured, state->parameter_generation_zero_fields.end());
  EXPECT_EQ(configured->config_source->target_operator_path, "child/op");
}

// REQ: PARAM-045
// REQ: PARAM-048
//   parent
//   |-- retained(ConfigBindingOperator)
//   `-- child(AttachedConfigGraph) [attempted]
//       `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, EmbeddingTwoGraphTransactionsRejectsWithoutMutation) {
  // Key setup: Attempt to embed a Graph transaction into a parent that already has one.
  Graph parent{"config-binding-embedded-transaction-conflict"};
  auto op = parent.op<ConfigBindingOperator>("retained");
  op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(33))}})},
  }));
  auto attached = set_params(parent, source, {bind(op, path("/managed"))});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  const GraphView before = parent.view();

  try {
    (void)parent.embed<AttachedConfigGraph>("child");
    FAIL() << "expected transaction conflict";
  } catch (const RuntimeError& error) {
    EXPECT_EQ(make_error(error).message, "GRAPH_PARAMETER_TRANSACTION_CONFLICT");
  }

  const GraphView after = parent.view();

  // Key assertion: The rejected embed leaves topology and the retained transaction unchanged.
  EXPECT_EQ(after.generation(), before.generation());
  EXPECT_EQ(after.identity(), before.identity());
  EXPECT_EQ(after.graphs().size(), before.graphs().size());
  EXPECT_EQ(after.vertices().size(), before.vertices().size());
  const ExecutionPlan plan = compile(parent);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const auto* compiled = static_cast<const ConfigBindingParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(compiled, nullptr);
  EXPECT_EQ(compiled->required, 33);
  EXPECT_EQ(compiled->optional, 9);
}

// REQ: PARAM-039
// REQ: PARAM-048
//   graph
//   `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, TryDecodeOwnsTypedValueAndUsesExplicitBaseline) {
  // Key setup: Decode a sparse config into an owning prepared value with an explicit baseline.
  auto decoded = [] {
    const config::Document source = document(object({
        {"managed",
         object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
    }));
    return try_decode<ConfigBindingParams>(source,
                                           path("/managed"),
                                           ConfigBindingParams{.required = 7, .optional = 9},
                                           config::DecodeOptions{});
  }();

  // Key assertion: The prepared value retains the baseline and can initialize an Operator.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  EXPECT_EQ(decoded->value().required, 42);
  EXPECT_EQ(decoded->value().optional, 9);
  EXPECT_EQ((**decoded).required, 42);
  EXPECT_EQ(decoded->operator->()->optional, 9);

  Graph graph{"prepared-params"};
  auto op = graph.op<ConfigBindingOperator>("op");
  op.set_params(std::move(*decoded));
  EXPECT_EQ(op->params().required, 42);
  EXPECT_EQ(op->params().optional, 9);

  const config::Document defaults_source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(11))}})},
  }));
  auto defaults =
      try_decode<ConfigBindingParams>(defaults_source, path("/managed"), config::DecodeOptions{});
  ASSERT_TRUE(defaults.has_value()) << defaults.error().token();
  ConfigBindingParams extracted = std::move(*defaults).take();
  EXPECT_EQ(extracted.required, 11);
  EXPECT_EQ(extracted.optional, 2);
}

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
// REQ: PARAM-048
// REQ: PARAM-058
TEST(ParameterConfigBinding, DecodeReturnsPreparedValue) {
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
  }));

  auto decoded = decode<ConfigBindingParams>(source, "/managed");

  EXPECT_EQ(decoded->required, 42);
  EXPECT_EQ(decoded->optional, 2);
}

// REQ: PARAM-024
// REQ: PARAM-058
TEST(ParameterConfigBinding, DecodeFormatsFailure) {
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_string("invalid"))}})},
  }));
  try {
    (void)decode<ConfigBindingParams>(source, "/managed");
    FAIL() << "expected decode failure";
  } catch (const ParamException& exception) {
    EXPECT_EQ(exception.code(), ErrorCode::kFailure);
    EXPECT_EQ(exception.error().code(), ParamToolErrorCode::kInvalidArgument);
    EXPECT_EQ(exception.error().token(), "PARAM_CONFIG_SCALAR_INVALID");
    ASSERT_TRUE(exception.error().config_cause().has_value());
    EXPECT_EQ(exception.error().config_cause()->code(), config::ConfigErrorCode::kTypeMismatch);
    EXPECT_EQ(exception.error().config_cause()->path(), "/managed/required");
    EXPECT_EQ(exception.what(), format_error(exception.error()));
  }
}
#endif

// REQ: PARAM-048
TEST(ParameterConfigBinding, TryDecodeReportsUnknownFieldSourcePath) {
  const config::Document source = document(object({
      {"unitree_g1_robot",
       object({{"unknown", value_or_null(config::ConfigValue::try_signed_integer(1))}})},
  }));

  auto decoded = try_decode<ConfigBindingParams>(source, "/unitree_g1_robot");

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_UNKNOWN");
  const auto& config_error = decoded.error().config_cause();
  ASSERT_TRUE(config_error.has_value());
  EXPECT_EQ(config_error->path(), "/unitree_g1_robot/unknown");
  EXPECT_NE(format_error(decoded.error()).find("/unitree_g1_robot/unknown"), std::string::npos);
}

// REQ: PARAM-048
TEST(ParameterConfigBinding, TryDecodeReportsInvalidFieldSourcePath) {
  const config::Document source = document(object({
      {"unitree_g1_robot",
       object({{"required", value_or_null(config::ConfigValue::try_string("invalid"))}})},
  }));

  auto decoded = try_decode<ConfigBindingParams>(source, "/unitree_g1_robot");

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_SCALAR_INVALID");
  const auto& config_error = decoded.error().config_cause();
  ASSERT_TRUE(config_error.has_value());
  EXPECT_EQ(config_error->path(), "/unitree_g1_robot/required");
  ASSERT_EQ(config_error->cause_count(), 1U);
  ASSERT_NE(config_error->cause(0), nullptr);
  EXPECT_EQ(config_error->cause(0)->code(), config::ConfigErrorCode::kTypeMismatch);
  EXPECT_NE(format_error(decoded.error()).find("/unitree_g1_robot/required"), std::string::npos);
}

// REQ: PARAM-048
TEST(ParameterConfigBinding, TryDecodeReportsUnavailableCodecSourcePath) {
  const config::Document source = document(object({
      {"managed",
       object({{"untouched", value_or_null(config::ConfigValue::try_signed_integer(8))}})},
  }));

  auto decoded = try_decode<AggregateCodecBindingParams>(source, "/managed");

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE");
  const auto& config_error = decoded.error().config_cause();
  ASSERT_TRUE(config_error.has_value());
  EXPECT_EQ(config_error->path(), "/managed/untouched");
}

// REQ: PARAM-048
TEST(ParameterConfigBinding, TryDecodeDoesNotAttachOversizedSectionPath) {
  const std::string key(512U, 'a');
  const config::Document source =
      document(object({{key, value_or_null(config::ConfigValue::try_signed_integer(1))}}));

  auto decoded = try_decode<ConfigBindingParams>(source, "/" + key);

  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_BINDING_OBJECT_REQUIRED");
  EXPECT_FALSE(decoded.error().config_cause().has_value());
}

// REQ: PARAM-039
TEST(ParameterConfigBinding, TryDecodeUsesFieldRoutesWhenAggregateCodecIsVisible) {
  // Key setup: Encode a nested schema value and decode it through its root field routes.
  const SchemaCodecInferenceParams source{.iterations = 7, .engine = "optimized"};
  auto encoded = config::try_encode_config(source);
  ASSERT_TRUE(encoded);
  auto source_document = config::Document::try_from_value(std::move(*encoded));
  ASSERT_TRUE(source_document);

  auto decoded = try_decode<SchemaCodecInferenceParams>(
      *source_document, config::Path::root(), config::DecodeOptions{});

  // Key assertion: Field-route decoding reconstructs the complete source value.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  EXPECT_EQ(decoded->value(), source);
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingDecodesVectorFieldWithoutAggregateCodec) {
  // Key setup: Decode only a vector field over a baseline without an aggregate codec.
  const ContainerBindingParams baseline{
      .labels = {"baseline"}, .gains = {{"retained", 3.0}}, .untouched = 11};
  const config::Document source = document(object({
      {"managed",
       object({{"labels",
                array({value_or_null(config::ConfigValue::try_string("left")),
                       value_or_null(config::ConfigValue::try_string("right"))})}})},
  }));

  auto decoded = try_decode<ContainerBindingParams>(source, "/managed", baseline);

  // Key assertion: The vector is replaced while unrelated baseline fields are retained.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  EXPECT_EQ(decoded->value().labels, (std::vector<std::string>{"left", "right"}));
  EXPECT_EQ(decoded->value().gains, (std::map<std::string, double>{{"retained", 3.0}}));
  EXPECT_EQ(decoded->value().untouched, 11);
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingReplacesMapFieldWithoutAggregateCodec) {
  // Key setup: Decode a replacement map over a baseline without an aggregate codec.
  const ContainerBindingParams baseline{.gains = {{"retained", 3.0}, {"removed", 4.0}}};
  const config::Document source = document(object({
      {"managed",
       object(
           {{"gains",
             object({{"replacement", value_or_null(config::ConfigValue::try_floating(2.5))}})}})},
  }));

  auto decoded = try_decode<ContainerBindingParams>(source, "/managed", baseline);

  // Key assertion: The map is replaced as a field while other defaults remain intact.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  EXPECT_EQ(decoded->value().gains, (std::map<std::string, double>{{"replacement", 2.5}}));
  EXPECT_EQ(decoded->value().labels, (std::vector<std::string>{"baseline"}));
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, TypedBindingComposesVectorWithVisibleElementCodec) {
  // Key setup: Decode a vector whose element type exposes its own config codec.
  const config::Document source = document(object({
      {"managed",
       object({{"history",
                array({value_or_null(config::ConfigValue::try_string("state")),
                       value_or_null(config::ConfigValue::try_string("action"))})}})},
  }));

  auto decoded = try_decode<ContainerBindingParams>(source, "/managed");

  // Key assertion: The vector codec composes the visible element codec for each entry.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  ASSERT_EQ(decoded->value().history.size(), 2U);
  EXPECT_EQ(decoded->value().history[0].input_name, "state");
  EXPECT_EQ(decoded->value().history[1].input_name, "action");
}

// REQ: PARAM-033
// REQ: PARAM-038
TEST(ParameterConfigBinding, TypedBindingDecodesNestedRequiredParamSchemaCodec) {
  // Key setup: Decode a complete and then an incomplete required nested schema object.
  const ContainerBindingParams baseline{.labels = {"baseline"},
                                        .inference = {.iterations = 3, .engine = "baseline"},
                                        .untouched = 11};
  const config::Document source = document(object({
      {"managed",
       object({{"inference",
                object({{"iterations", value_or_null(config::ConfigValue::try_signed_integer(5))},
                        {"engine_name",
                         value_or_null(config::ConfigValue::try_string("configured"))}})}})},
  }));

  auto decoded = try_decode<ContainerBindingParams>(source, "/managed", baseline);

  // Key assertion: Complete input replaces the nested value; missing required input rejects.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  EXPECT_EQ(decoded->value().inference,
            (SchemaCodecInferenceParams{.iterations = 5, .engine = "configured"}));
  EXPECT_EQ(decoded->value().labels, (std::vector<std::string>{"baseline"}));
  EXPECT_EQ(decoded->value().untouched, 11);

  const config::Document partial = document(object({
      {"managed",
       object({{"inference",
                object({{"iterations",
                         value_or_null(config::ConfigValue::try_signed_integer(8))}})}})},
  }));
  auto missing_required = try_decode<ContainerBindingParams>(partial, "/managed", baseline);

  ASSERT_FALSE(missing_required.has_value());
  EXPECT_EQ(missing_required.error().token(), "PARAM_CONFIG_FIELD_DECODE_FAILED");
}

// REQ: PARAM-033
// REQ: PARAM-045
// REQ: PARAM-048
//   graph
//   `-- op(AggregateCodecBindingOperator)
TEST(ParameterConfigBinding, GraphAttachmentRejectsAggregateOnlyFieldCodec) {
  // Key setup: Bind a field whose type provides only an aggregate codec.
  Graph graph{"config-binding-aggregate-only"};
  auto op = graph.op<AggregateCodecBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"untouched", value_or_null(config::ConfigValue::try_signed_integer(8))}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: Graph attachment requires a field codec and rejects atomically.
  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().token(), "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE");
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, InvalidContainerFieldReturnsNoValue) {
  // Key setup: Supply an integer element for a string-vector field.
  const config::Document source = document(object({
      {"managed",
       object({{"labels", array({value_or_null(config::ConfigValue::try_signed_integer(1))})}})},
  }));

  auto decoded = try_decode<ContainerBindingParams>(source, "/managed");

  // Key assertion: The invalid element makes the entire typed decode fail.
  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_DECODE_FAILED");
}

// REQ: PARAM-033
//   graph
//   `-- op(UnsupportedBindingOperator)
TEST(ParameterConfigBinding, TypedBindingReportsUnavailableFieldCodec) {
  // Key setup: Bind a field whose unsupported value type has no field codec.
  Graph graph{"config-binding-unsupported-field"};
  auto op = graph.op<UnsupportedBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"value",
                object({{"number", value_or_null(config::ConfigValue::try_signed_integer(1))}})}})},
  }));

  auto attached = set_params(graph, source, {bind(op, path("/managed"))});

  // Key assertion: The attachment reports codec unavailability without partial state.
  ASSERT_FALSE(attached.has_value());
  EXPECT_EQ(attached.error().token(), "PARAM_CONFIG_FIELD_CODEC_UNAVAILABLE");
}

// REQ: PARAM-033
//   graph
//   `-- op(ConfigBindingOperator)
TEST(ParameterConfigBinding, CanonicalAndAliasContributionRejectAtomically) {
  // Key setup: Contribute both a canonical field name and its alias to one Operator.
  Graph graph{"config-binding-duplicates"};
  auto op = graph.op<ConfigBindingOperator>("op");
  const config::Document source = document(object({
      {"managed",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))},
               {"legacy_required", value_or_null(config::ConfigValue::try_signed_integer(41))}})},
  }));
  const std::array bindings{bind(op, path("/managed"))};

  auto decoded = set_params(graph, source, bindings);

  // Key assertion: Duplicate semantic fields reject the complete attachment.
  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_DUPLICATE");
}

// REQ: PARAM-033
//   graph
//   `-- camera/front(ConfigBindingOperator)
// JSON Pointer "/profiles/camera~1front/1" uses RFC 6901 escaping for the slash in the key name.
TEST(ParameterConfigBinding, BindingKeepsGraphAndJsonPointerGrammarsSeparate) {
  // Key setup: Bind a slash-containing Operator through an independently escaped JSON Pointer.
  Graph graph{"dynamic-config-binding"};
  auto op = graph.op<ConfigBindingOperator>("camera/front");
  op.set_params(ConfigBindingParams{.required = 7, .optional = 9});
  const config::Document source = document(object({
      {"profiles",
       object({{"camera/front",
                array({config::ConfigValue::null(),
                       object({{"required",
                                value_or_null(config::ConfigValue::try_signed_integer(43))}})})}})},
  }));
  const std::array bindings{bind(op, path("/profiles/camera~1front/1"))};

  auto decoded = set_params(graph, source, bindings);

  // Key assertion: Operator paths use percent encoding independently of JSON Pointer escaping.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  ASSERT_EQ(decoded->config_fields.size(), 1U);
  EXPECT_EQ(decoded->config_fields.front().target_operator_path, "camera%2Ffront");
}

// REQ: PARAM-033
//   graph
//   |-- first(ConfigBindingOperator)
//   `-- second(ConfigBindingOperator)
TEST(ParameterConfigBinding, MultiOperatorAttachmentReturnsNoPartialTransaction) {
  // Key setup: Bind two Operators when the second source contains an unknown field.
  Graph graph{"multi-config-binding"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ConfigBindingOperator>("second");
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(20))}})},
      {"second", object({{"unknown", value_or_null(config::ConfigValue::try_signed_integer(30))}})},
  }));
  const std::array bindings{bind(first, path("/first")), bind(second, path("/second"))};

  auto decoded = set_params(graph, source, bindings);

  // Key assertion: One invalid target rejects the transaction without a partial attachment.
  ASSERT_FALSE(decoded.has_value());
  EXPECT_EQ(decoded.error().token(), "PARAM_CONFIG_FIELD_UNKNOWN");
}

// REQ: PARAM-033
//   graph
//   |-- first(ConfigBindingOperator)
//   `-- second(ConfigBindingOperator)
TEST(ParameterConfigBinding, DecodeLimitsDistinguishInvalidOptionsFromCapacityPressure) {
  // Key setup: Compare an invalid zero field limit with an exhausted one-binding limit.
  Graph graph{"config-binding-limits"};
  auto first = graph.op<ConfigBindingOperator>("first");
  auto second = graph.op<ConfigBindingOperator>("second");
  const config::Document source = document(object({
      {"first", object({{"required", value_or_null(config::ConfigValue::try_signed_integer(20))}})},
      {"second",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(30))}})},
  }));

  ParamConfigDecodeOptions invalid;
  invalid.max_fields = 0U;
  auto invalid_result = set_params(graph, source, {bind(first, path("/first"))}, invalid);
  ASSERT_FALSE(invalid_result.has_value());
  EXPECT_EQ(invalid_result.error().code(), ParamToolErrorCode::kInvalidArgument);
  EXPECT_EQ(invalid_result.error().token(), "PARAM_CONFIG_OPTIONS_INVALID");

  ParamConfigDecodeOptions constrained;
  constrained.max_bindings = 1U;
  auto exhausted = set_params(
      graph, source, {bind(first, path("/first")), bind(second, path("/second"))}, constrained);

  // Key assertion: Invalid options and capacity exhaustion return distinct error categories.
  ASSERT_FALSE(exhausted.has_value());
  EXPECT_EQ(exhausted.error().code(), ParamToolErrorCode::kResourceExhausted);
  EXPECT_EQ(exhausted.error().token(), "PARAM_CONFIG_BINDING_CAPACITY");
}

// Builds a synthetic ParameterCatalogView for a single "required" field on operator "op".
// Wires external_access=kTuning and range constraints from the schema descriptor so that
// RuntimeDecodeUsesRestrictedCatalogAndConstraints can exercise out-of-range rejection (>100).
[[nodiscard]] ParameterCatalogView runtime_catalog() {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<ConfigBindingParams>();
  const ParamFieldDescriptor& descriptor = schema->fields.front();
  auto operator_path = OperatorPath::parse("op");
  auto field_path = ParamPath::parse("required");
  EXPECT_TRUE(operator_path.has_value());
  EXPECT_TRUE(field_path.has_value());
  const OperatorRef target{.id = 1U, .plan_generation = 2U, .run_generation = 3U};
  OperatorParamRecord op{
      .operator_instance = target,
      .path = operator_path ? std::move(*operator_path) : OperatorPath{},
      .effective_profile = Profile::kStandard,
      .schema_availability = ParamSchemaAvailability::kDeclared,
      .schema_disposition = ParameterSchemaAdmissionDisposition::kCompleteProven,
      .materialization = ParameterMaterializationStrategy::kFixedObject,
      .schema_id = std::string{schema->stable_id},
      .schema_version = schema->version,
      .structural_hash = schema->structural_hash,
      .contract_hash = schema->contract_hash,
      .metadata_hash = schema->metadata_hash,
      .schema_coverage = schema->coverage,
      .schema_proof = schema->proof,
  };
  ParamFieldCatalogRecord field{
      .instance = 1U,
      .address = ParamAddress{.operator_instance = target, .field = descriptor.id},
      .full_id = descriptor.full_id,
      .path = field_path ? std::move(*field_path) : ParamPath{},
      .logical_type = descriptor.logical_type,
      .type_shape = descriptor.type_shape,
      .mutability = ParamMutability::kRuntime,
      .persistence = descriptor.persistence,
      .visibility = ParamVisibility::kPublic,
      .external_access = ParamExternalAccess::kTuning,
      .sensitivity = ParamSensitivity::kNormal,
      .constraints = descriptor.constraints,
      .value_route = ParamValueRouteDisposition::kPortableScalar,
      .materialization = ParameterMaterializationStrategy::kFixedObject,
      .projection = ParamProjectionStatus::kIncluded,
      .update_disposition = ParameterFieldUpdateDisposition::kAdmitted,
      .adapter_eligible = true,
  };
  ParamCatalogDigest digest{};
  digest.value.front() = std::byte{1U};
  return holoscan::detail::ParameterCatalogAccess::make(
      digest,
      std::vector<OperatorParamRecord>{std::move(op)},
      std::vector<ParamFieldCatalogRecord>{std::move(field)});
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, RuntimeDecodeUsesRestrictedCatalogAndConstraints) {
  // Key setup: Decode one valid and one out-of-range patch through a restricted catalog.
  const auto catalog = runtime_catalog();
  const config::Document source = document(object({
      {"values",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(42))}})},
      {"generation", value_or_null(config::ConfigValue::try_unsigned_integer(7U))},
  }));
  auto operator_path = OperatorPath::parse("op");
  ASSERT_TRUE(operator_path.has_value());
  const RuntimeParamConfigBinding binding{
      .operator_path = std::move(*operator_path),
      .values_path = path("/values"),
      .expected_generation_path = path("/generation"),
  };

  auto decoded = decode_runtime_patches(source, catalog, binding);

  // Key assertion: The valid patch carries its generation; the constrained value rejects.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  ASSERT_EQ(decoded->size(), 1U);
  EXPECT_EQ(decoded->front().expected_generation, 7U);
  auto value = decoded->front().value.as<std::int32_t>();
  ASSERT_TRUE(value.has_value());
  EXPECT_EQ(*value, 42);

  const config::Document invalid = document(object({
      {"values",
       object({{"required", value_or_null(config::ConfigValue::try_signed_integer(101))}})},
      {"generation", value_or_null(config::ConfigValue::try_unsigned_integer(7U))},
  }));
  auto rejected = decode_runtime_patches(invalid, catalog, binding);
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().token(), "PARAM_CONFIG_FIELD_CONSTRAINT");
}

// REQ: PARAM-033
TEST(ParameterConfigBinding, SnapshotProjectionProducesIncludedFieldMergeOnly) {
  // Key setup: Build a snapshot with one included and one transiently excluded field.
  auto operator_path = OperatorPath::parse("op");
  auto required_path = ParamPath::parse("required");
  auto optional_path = ParamPath::parse("optional");
  ASSERT_TRUE(operator_path.has_value());
  ASSERT_TRUE(required_path.has_value());
  ASSERT_TRUE(optional_path.has_value());
  std::vector<ParamSnapshotFieldRecord> fields;
  fields.push_back(ParamSnapshotFieldRecord{.path = std::move(*required_path),
                                            .projection = ParamProjectionStatus::kIncluded,
                                            .value = ParamValue::from(std::int32_t{42})});
  fields.push_back(ParamSnapshotFieldRecord{
      .path = std::move(*optional_path),
      .projection = ParamProjectionStatus::kExcludedTransient,
  });
  ParamSnapshotOperatorRecord op{
      .path = std::move(*operator_path),
      .fields = std::move(fields),
  };
  const ParamSnapshot snapshot = holoscan::detail::ParamSnapshotArchiveAccess::snapshot(
      {},
      {},
      {},
      {},
      ParamReadView::kPublished,
      ParamCaptureConsistency::kRequireStopped,
      ParamSnapshotScope::root(),
      std::nullopt,
      std::nullopt,
      std::vector<ParamSnapshotOperatorRecord>{std::move(op)});

  auto edit = to_config_edit(snapshot, path("/managed"));

  // Key assertion: Projection emits only the included field in the merge edit.
  ASSERT_TRUE(edit.has_value()) << edit.error().token();
  EXPECT_EQ(edit->kind(), config::EditKind::kMergeObject);
  auto required = edit->value().view().try_member("required");
  ASSERT_TRUE(required.has_value());
  auto decoded = required->try_as_signed_integer();
  ASSERT_TRUE(decoded.has_value());
  EXPECT_EQ(*decoded, 42);
  auto optional = edit->value().view().try_find_member("optional");
  ASSERT_TRUE(optional.has_value());
  EXPECT_FALSE(optional->has_value());
}

}  // namespace
}  // namespace holoscan::params
