/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/config/codec.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/params/config.hpp>

#include "../support/optional_assertions.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/parameter_catalog_internal.hpp"
#include "core/parameter_input_internal.hpp"

namespace holoscan::params::test {

// Params schema marked `.partial()`: only "managed" is schema-tracked; "local_only" is a typed
// baseline field invisible to the parameter system. The companion ConfigCodec encodes/decodes
// both fields so round-trips preserve the full object. Representative test:
// PartialSchemaOverlayPreservesTypedUnmanagedBaseline.
struct PartialCodecParams {
  std::string managed{"typed-default"};
  std::string local_only{"local-default"};
};

HOLOSCAN_PARAM_SCHEMA(
    PartialCodecParams, param_schema("holoscan.test.PartialCodecParams", 1).partial(),
    field<&PartialCodecParams::managed>("managed").encoding(ParamStringEncoding::kUtf8));

// No-op operator carrying PartialCodecParams. Used to register operators in provenance test
// graphs without any real compute logic.
class PartialCodecOperator final : public Operator<PartialCodecParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

struct ConfigProvenancePairParams {
  std::uint32_t primary{1U};
  std::uint32_t secondary{2U};
};

HOLOSCAN_PARAM_SCHEMA(ConfigProvenancePairParams,
                      param_schema("holoscan.test.ConfigProvenancePairParams", 1),
                      field<&ConfigProvenancePairParams::primary>("primary"),
                      field<&ConfigProvenancePairParams::secondary>("secondary"));

class ConfigProvenancePairOperator final : public Operator<ConfigProvenancePairParams> {
 public:
  void setup(OperatorSpec&) override {}
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

}  // namespace holoscan::params::test

// Custom full-object codec: decodes/encodes both "managed" and "local_only" from the config
// document even though only "managed" is registered in HOLOSCAN_PARAM_SCHEMA. This simulates
// an operator that owns fields the parameter system cannot see.
template <>
struct holoscan::config::ConfigCodec<holoscan::params::test::PartialCodecParams> {
  static ConfigResult<holoscan::params::test::PartialCodecParams> decode(
      ConfigView view, const DecodeOptions& options, DecodeContext& context) {
    ObjectDecoder object{std::move(view), options, context};
    auto managed = object.required<std::string>("managed");
    if (!managed) {
      return tl::make_unexpected(managed.error());
    }
    auto local_only = object.required<std::string>("local_only");
    if (!local_only) {
      return tl::make_unexpected(local_only.error());
    }
    auto finished = object.finish();
    if (!finished) {
      return tl::make_unexpected(finished.error());
    }
    return holoscan::params::test::PartialCodecParams{.managed = std::move(*managed),
                                                      .local_only = std::move(*local_only)};
  }

  static ConfigResult<ConfigValue> encode(const holoscan::params::test::PartialCodecParams& value,
                                          const EncodeOptions& options, EncodeContext& context) {
    auto managed = context.encode_child<std::string>("managed", value.managed, options);
    if (!managed) {
      return tl::make_unexpected(managed.error());
    }
    auto local_only = context.encode_child<std::string>("local_only", value.local_only, options);
    if (!local_only) {
      return tl::make_unexpected(local_only.error());
    }
    const std::array entries{ConfigObjectEntry{"managed", *managed},
                             ConfigObjectEntry{"local_only", *local_only}};
    return ConfigValue::try_object(entries);
  }
};

namespace holoscan::params::test {
namespace {

// Test DSL helper: parse_json wraps config_io::parse with an assertion and a null fallback.
[[nodiscard]] config::Document parse_json(std::string_view source) {
  auto parsed = config_io::try_parse(source, config_io::Format::kJson);
  EXPECT_TRUE(parsed.has_value());
  if (parsed) {
    return std::move(*parsed);
  }
  return config::Document::try_from_value(config::ConfigValue::null()).value();
}

// Test DSL shorthand for paths whose literals are valid by construction.
[[nodiscard]] config::Path path(std::string_view value) {
  return config::Path::parse(value);
}

class PreparedParamsChildGraph final : public Graph {
 protected:
  void compose() override {
    const config::Document source = parse_json(R"({"primary":9})");
    auto decoded = try_decode<ConfigProvenancePairParams>(
        source,
        config::Path::root(),
        ConfigProvenancePairParams{.primary = 3U, .secondary = 7U},
        config::DecodeOptions{});
    if (!decoded) {
      throw std::runtime_error{std::string{decoded.error().token()}};
    }
    auto configured = op<ConfigProvenancePairOperator>("op");
    configured.set_params(std::move(*decoded));
  }
};

// Returns true if any diagnostic in the plan carries the given token string.
[[nodiscard]] bool has_diagnostic(const ExecutionPlan& plan, std::string_view token) {
  return std::ranges::any_of(plan.diagnostics(),
                             [&](const Diagnostic& item) { return item.token == token; });
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

[[nodiscard]] const ConfigProvenancePairParams* compiled_pair_params(const ExecutionPlan& plan) {
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  if (state == nullptr || state->executable_factories.operators.empty() ||
      !state->executable_factories.operators.front().factory) {
    return nullptr;
  }
  return static_cast<const ConfigProvenancePairParams*>(
      state->executable_factories.operators.front().factory->initial_parameter_value());
}

// Builds a synthetic ParameterCatalogView for a single "managed" field on operator "op".
// Parameterised on external_access so tests can flip between kTuning (adapter-eligible)
// and kNone (blocked) to exercise the adapter-eligibility gate in decode_runtime_patches.
[[nodiscard]] ParameterCatalogView runtime_catalog(ParamExternalAccess external_access) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<PartialCodecParams>();
  const ParamFieldDescriptor& descriptor = schema->fields.front();
  auto operator_path = OperatorPath::parse("op");
  auto field_path = ParamPath::parse("managed");
  EXPECT_TRUE(operator_path.has_value());
  EXPECT_TRUE(field_path.has_value());
  const OperatorRef target{.id = 1U, .plan_generation = 2U, .run_generation = 3U};
  constexpr auto value_route = ParamValueRouteDisposition::kPortableScalar;
  constexpr auto projection = ParamProjectionStatus::kIncluded;
  OperatorParamRecord op{
      .operator_instance = target,
      .path = operator_path ? std::move(*operator_path) : OperatorPath{},
      .effective_profile = Profile::kStandard,
      .schema_availability = ParamSchemaAvailability::kDeclared,
      .schema_disposition = ParameterSchemaAdmissionDisposition::kDeclaredPartial,
      .materialization = ParameterMaterializationStrategy::kColdOwnedTyped,
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
      .external_access = external_access,
      .sensitivity = ParamSensitivity::kNormal,
      .constraints = descriptor.constraints,
      .value_route = value_route,
      .materialization = ParameterMaterializationStrategy::kFixedObject,
      .projection = projection,
      .update_disposition = ParameterFieldUpdateDisposition::kAdmitted,
      .adapter_eligible = projection == ParamProjectionStatus::kIncluded &&
                          external_access != ParamExternalAccess::kNone &&
                          value_route == ParamValueRouteDisposition::kPortableScalar,
  };
  ParamCatalogDigest digest{};
  digest.value.front() = std::byte{1U};
  return holoscan::detail::ParameterCatalogAccess::make(
      digest,
      std::vector<OperatorParamRecord>{std::move(op)},
      std::vector<ParamFieldCatalogRecord>{std::move(field)});
}

// REQ: COMPILE-072
// REQ: PARAM-034
// REQ: PARAM-037
// REQ: PARAM-039
// Graph: graph["partial-config-binding"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, PartialSchemaOverlayPreservesTypedUnmanagedBaseline) {
  // Key setup: baseline has managed="typed-managed", local_only="keep-local"; JSON document
  // provides only managed="configured". The partial codec must keep local_only from the baseline.
  Graph graph{"partial-config-binding"};
  auto op = graph.op<PartialCodecOperator>("op");
  op.set_params(PartialCodecParams{.managed = "typed-managed", .local_only = "keep-local"});
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");

  auto attached = set_params(graph, source, {bind(op, "/overlay")});

  // Key assertion: managed field is patched; local_only is preserved; provenance records
  // the exact source path, semantic digest, and raw digest of the config document.
  ASSERT_TRUE(attached.has_value()) << attached.error().token();
  ASSERT_EQ(attached->config_fields.size(), 1U);
  const auto& source_field = attached->config_fields.front();
  const auto source_digest = source.digest();
  EXPECT_EQ(source_field.document.semantic_digest, source_digest.semantic.sha256);
  const auto& source_identity = source_digest.source;
  HOLOSCAN_ASSERT_HAS_VALUE(source_identity);
  const auto& raw_source_digest = *source_identity;
  EXPECT_TRUE(source_field.document.has_raw_source);
  EXPECT_EQ(source_field.document.raw_digest, raw_source_digest.sha256);
  EXPECT_FALSE(source_field.document.has_base_raw_source);
  EXPECT_EQ(source_field.source_path, "/overlay/managed");
  EXPECT_EQ(source_field.source_ordinal, 0U);
  EXPECT_EQ(source_field.target_operator_path, "op");
  EXPECT_EQ(source_field.target_schema_id, "holoscan.test.PartialCodecParams");

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* plan_state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(plan_state, nullptr);
  const auto field =
      std::ranges::find_if(plan_state->parameter_generation_zero_fields,
                           [](const auto& item) { return item.config_source.has_value(); });
  ASSERT_NE(field, plan_state->parameter_generation_zero_fields.end());
  ASSERT_TRUE(field->config_source.has_value());
  const auto config_source = field->config_source.value_or(ParamConfigFieldProvenance{});
  EXPECT_EQ(config_source.source_path, "/overlay/managed");
  EXPECT_EQ(config_source.document.raw_digest, raw_source_digest.sha256);
  const auto* value = static_cast<const PartialCodecParams*>(
      plan_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->managed, "configured");
  EXPECT_EQ(value->local_only, "keep-local");
}

// REQ: PARAM-039
// REQ: PARAM-048
TEST(ParameterConfigProvenance, PreparedParamsRetainOnlySuppliedFieldEvidence) {
  Graph graph{"prepared-config-provenance"};
  auto op = graph.op<ConfigProvenancePairOperator>("op");
  const config::Document source = parse_json(R"({"input":{"primary":9}})");
  auto decoded = try_decode<ConfigProvenancePairParams>(
      source,
      path("/input"),
      ConfigProvenancePairParams{.primary = 3U, .secondary = 7U},
      config::DecodeOptions{});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  op.set_params(std::move(*decoded));

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->parameter_generation_zero_fields.size(), 2U);
  const auto configured =
      std::ranges::find_if(state->parameter_generation_zero_fields,
                           [](const auto& field) { return field.config_source.has_value(); });
  ASSERT_NE(configured, state->parameter_generation_zero_fields.end());
  const auto& config_source = configured->config_source;
  HOLOSCAN_ASSERT_HAS_VALUE(config_source);
  EXPECT_EQ(configured->origin, holoscan::detail::ParameterGenerationZeroOrigin::kInputCaptured);
  EXPECT_EQ(config_source->source_path, "/input/primary");
  EXPECT_EQ(config_source->target_operator_path, "op");
  const auto baseline =
      std::ranges::find_if(state->parameter_generation_zero_fields,
                           [](const auto& field) { return !field.config_source.has_value(); });
  ASSERT_NE(baseline, state->parameter_generation_zero_fields.end());
  EXPECT_EQ(baseline->origin, holoscan::detail::ParameterGenerationZeroOrigin::kTypedBaseline);
}

// REQ: PARAM-039
// REQ: PARAM-048
// REQ: COMPILE-073
TEST(ParameterConfigProvenance, PreparedAndPlainValuesShareSemanticPlanIdentity) {
  Graph prepared_graph{"equivalent-inputs"};
  auto prepared_op = prepared_graph.op<ConfigProvenancePairOperator>("op");
  const config::Document source = parse_json(R"({ "input" : { "primary" : 9 } })");
  auto decoded = try_decode<ConfigProvenancePairParams>(
      source,
      path("/input"),
      ConfigProvenancePairParams{.primary = 3U, .secondary = 7U},
      config::DecodeOptions{});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  prepared_op.set_params(std::move(*decoded));
  const ExecutionPlan prepared_plan = compile(prepared_graph);
  ASSERT_TRUE(prepared_plan.ok()) << prepared_plan.json();

  Graph plain_graph{"equivalent-inputs"};
  auto plain_op = plain_graph.op<ConfigProvenancePairOperator>("op");
  plain_op.set_params(ConfigProvenancePairParams{.primary = 9U, .secondary = 7U});
  const ExecutionPlan plain_plan = compile(plain_graph);
  ASSERT_TRUE(plain_plan.ok()) << plain_plan.json();

  const auto* prepared_state = holoscan::detail::ExecutionPlanAccess::state(prepared_plan);
  const auto* plain_state = holoscan::detail::ExecutionPlanAccess::state(plain_plan);
  ASSERT_NE(prepared_state, nullptr);
  ASSERT_NE(plain_state, nullptr);
  EXPECT_EQ(prepared_state->plan_generation, plain_state->plan_generation);

  prepared_op.set_params(ConfigProvenancePairParams{.primary = 9U, .secondary = 7U});
  const ExecutionPlan replaced_plan = compile(prepared_graph);
  ASSERT_TRUE(replaced_plan.ok()) << replaced_plan.json();
  const auto* replaced_state = holoscan::detail::ExecutionPlanAccess::state(replaced_plan);
  ASSERT_NE(replaced_state, nullptr);
  EXPECT_TRUE(
      std::ranges::none_of(replaced_state->parameter_generation_zero_fields,
                           [](const auto& field) { return field.config_source.has_value(); }));
}

// REQ: PARAM-039
// REQ: COMPILE-073
TEST(ParameterConfigProvenance, PreparedScalarValuesChangeSemanticPlanIdentity) {
  const auto compile_value = [](std::string_view json) {
    Graph graph{"different-inputs"};
    auto op = graph.op<ConfigProvenancePairOperator>("op");
    const config::Document source = parse_json(json);
    auto decoded = try_decode<ConfigProvenancePairParams>(
        source,
        path("/input"),
        ConfigProvenancePairParams{.primary = 3U, .secondary = 7U},
        config::DecodeOptions{});
    EXPECT_TRUE(decoded.has_value());
    if (!decoded) {
      return PlanGenerationId{};
    }
    op.set_params(std::move(*decoded));
    const ExecutionPlan plan = compile(graph);
    EXPECT_TRUE(plan.ok()) << plan.json();
    const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
    EXPECT_NE(state, nullptr);
    return state == nullptr ? PlanGenerationId{} : state->plan_generation;
  };

  EXPECT_NE(compile_value(R"({"input":{"primary":9}})"),
            compile_value(R"({"input":{"primary":10}})"));
}

// REQ: PARAM-039
TEST(ParameterConfigProvenance, EmbeddedPreparedParamsRetainFinalOperatorPath) {
  Graph parent{"embedded-prepared-params"};
  (void)parent.embed<PreparedParamsChildGraph>("child");

  const ExecutionPlan plan = compile(parent);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto configured =
      std::ranges::find_if(state->parameter_generation_zero_fields,
                           [](const auto& field) { return field.config_source.has_value(); });
  ASSERT_NE(configured, state->parameter_generation_zero_fields.end());
  const auto& config_source = configured->config_source;
  HOLOSCAN_ASSERT_HAS_VALUE(config_source);
  EXPECT_EQ(config_source->target_operator_path, "child/op");
}

// REQ: PARAM-053
// REQ: PARAM-048
// REQ: COMPILE-073
TEST(ParameterConfigProvenance, PreparedEvidenceRejectsZeroCurrentRawSourceVersion) {
  Graph graph{"prepared-zero-current-raw-version"};
  auto op = graph.op<ConfigProvenancePairOperator>("op");
  const config::Document source = parse_json(R"({"input":{"primary":9}})");
  auto decoded = try_decode<ConfigProvenancePairParams>(
      source,
      path("/input"),
      ConfigProvenancePairParams{.primary = 3U, .secondary = 7U},
      config::DecodeOptions{});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& source_evidence = holoscan::detail::PreparedParamsAccess::evidence(*decoded);
  ASSERT_NE(source_evidence, nullptr);
  auto evidence = std::make_shared<holoscan::detail::ParameterInputEvidenceState>(*source_evidence);
  ASSERT_EQ(evidence->config_fields.size(), 1U);
  ASSERT_TRUE(evidence->config_fields.front().document.has_raw_source);
  evidence->config_fields.front().document.raw_algorithm_version = 0U;
  auto forged = holoscan::detail::PreparedParamsAccess::make(
      std::move(*decoded).take(),
      std::shared_ptr<const holoscan::detail::ParameterInputEvidenceState>{std::move(evidence)});
  op.set_params(std::move(forged));

  const ExecutionPlan plan = compile(graph);

  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH"));
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(std::ranges::none_of(state->parameter_generation_zero_fields, [](const auto& field) {
    return field.config_source.has_value();
  }));
}

// REQ: PARAM-053
// REQ: PARAM-048
// REQ: COMPILE-073
TEST(ParameterConfigProvenance, PreparedEvidenceRejectsZeroBaseRawSourceVersion) {
  Graph graph{"prepared-zero-base-raw-version"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"before"}})");
  auto derived = source.try_replace("/overlay/managed", std::string{"configured"});
  ASSERT_TRUE(derived.has_value());
  ASSERT_FALSE(derived->digest().source.has_value());
  ASSERT_TRUE(derived->digest().base_source.has_value());
  auto decoded = try_decode<PartialCodecParams>(*derived, "/overlay");
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& source_evidence = holoscan::detail::PreparedParamsAccess::evidence(*decoded);
  ASSERT_NE(source_evidence, nullptr);
  auto evidence = std::make_shared<holoscan::detail::ParameterInputEvidenceState>(*source_evidence);
  ASSERT_EQ(evidence->config_fields.size(), 1U);
  ASSERT_TRUE(evidence->config_fields.front().document.has_base_raw_source);
  evidence->config_fields.front().document.base_raw_algorithm_version = 0U;
  auto forged = holoscan::detail::PreparedParamsAccess::make(
      std::move(*decoded).take(),
      std::shared_ptr<const holoscan::detail::ParameterInputEvidenceState>{std::move(evidence)});
  op.set_params(std::move(forged));

  const ExecutionPlan plan = compile(graph);

  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH"));
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  EXPECT_TRUE(std::ranges::none_of(state->parameter_generation_zero_fields, [](const auto& field) {
    return field.config_source.has_value();
  }));
}

// REQ: PARAM-053
// REQ: PARAM-048
// REQ: COMPILE-073
TEST(ParameterConfigProvenance, TransactionEvidenceRejectsZeroCurrentRawSourceVersion) {
  Graph donor{"transaction-zero-current-raw-donor"};
  auto donor_op = donor.op<ConfigProvenancePairOperator>("op");
  const config::Document source = parse_json(R"({"input":{"primary":19}})");
  auto staged = set_params(donor, source, {bind(donor_op, "/input")});
  ASSERT_TRUE(staged.has_value()) << staged.error().token();
  auto transaction = copy_parameter_transaction(donor);
  ASSERT_NE(transaction, nullptr);
  ASSERT_EQ(transaction->provenance.config_fields.size(), 1U);
  ASSERT_TRUE(transaction->provenance.config_fields.front().document.has_raw_source);
  transaction->provenance.config_fields.front().document.raw_algorithm_version = 0U;

  Graph graph{"transaction-zero-current-raw-target"};
  auto op = graph.op<ConfigProvenancePairOperator>("op");
  op.set_params(ConfigProvenancePairParams{.primary = 3U, .secondary = 7U});
  auto attached = holoscan::detail::ParameterInputAccess::attach_transaction(
      graph,
      std::shared_ptr<const holoscan::detail::ParameterTransactionState>{std::move(transaction)});
  ASSERT_TRUE(attached.has_value()) << attached.error().message;

  const ExecutionPlan plan = compile(graph);

  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH"));
  const ConfigProvenancePairParams* const resolved = compiled_pair_params(plan);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->primary, 3U);
  EXPECT_EQ(resolved->secondary, 7U);
}

// REQ: PARAM-053
// REQ: PARAM-048
// REQ: COMPILE-073
TEST(ParameterConfigProvenance, TransactionEvidenceRejectsZeroBaseRawSourceVersion) {
  Graph donor{"transaction-zero-base-raw-donor"};
  auto donor_op = donor.op<ConfigProvenancePairOperator>("op");
  const config::Document source = parse_json(R"({"input":{"primary":19}})");
  auto derived = source.try_replace("/input/primary", std::uint32_t{23U});
  ASSERT_TRUE(derived.has_value());
  ASSERT_FALSE(derived->digest().source.has_value());
  ASSERT_TRUE(derived->digest().base_source.has_value());
  auto staged = set_params(donor, *derived, {bind(donor_op, "/input")});
  ASSERT_TRUE(staged.has_value()) << staged.error().token();
  auto transaction = copy_parameter_transaction(donor);
  ASSERT_NE(transaction, nullptr);
  ASSERT_EQ(transaction->provenance.config_fields.size(), 1U);
  ASSERT_TRUE(transaction->provenance.config_fields.front().document.has_base_raw_source);
  transaction->provenance.config_fields.front().document.base_raw_algorithm_version = 0U;

  Graph graph{"transaction-zero-base-raw-target"};
  auto op = graph.op<ConfigProvenancePairOperator>("op");
  op.set_params(ConfigProvenancePairParams{.primary = 3U, .secondary = 7U});
  auto attached = holoscan::detail::ParameterInputAccess::attach_transaction(
      graph,
      std::shared_ptr<const holoscan::detail::ParameterTransactionState>{std::move(transaction)});
  ASSERT_TRUE(attached.has_value()) << attached.error().message;

  const ExecutionPlan plan = compile(graph);

  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH"));
  const ConfigProvenancePairParams* const resolved = compiled_pair_params(plan);
  ASSERT_NE(resolved, nullptr);
  EXPECT_EQ(resolved->primary, 3U);
  EXPECT_EQ(resolved->secondary, 7U);
}

// REQ: PARAM-053
// Graph: graph["derived-config-provenance"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, DerivedConfigRetainsBaseButNotExactSourceIdentity) {
  Graph graph{"derived-config-provenance"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"before"}})");
  const auto source_digest = source.digest();
  const auto& source_identity = source_digest.source;
  HOLOSCAN_ASSERT_HAS_VALUE(source_identity);

  auto derived = source.try_replace("/overlay/managed", std::string{"configured"});
  ASSERT_TRUE(derived);
  ASSERT_FALSE(derived->digest().source.has_value());
  ASSERT_EQ(derived->digest().base_source, source_digest.source);

  auto decoded = try_decode<PartialCodecParams>(*derived, "/overlay");
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  op.set_params(std::move(*decoded));
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  const auto configured =
      std::ranges::find_if(state->parameter_generation_zero_fields,
                           [](const auto& field) { return field.config_source.has_value(); });
  ASSERT_NE(configured, state->parameter_generation_zero_fields.end());
  const auto& config_source = configured->config_source;
  HOLOSCAN_ASSERT_HAS_VALUE(config_source);
  const auto& document = config_source->document;
  EXPECT_FALSE(document.has_raw_source);
  EXPECT_TRUE(document.has_base_raw_source);
  EXPECT_EQ(document.base_raw_algorithm_version, source_identity->algorithm_version);
  EXPECT_EQ(document.base_raw_digest, source_identity->sha256);
  EXPECT_EQ(document.base_raw_byte_size, source_identity->byte_size);
}

// REQ: COMPILE-073
TEST(ParameterConfigProvenance, ConfigInputUsesTheActualValidCompileProfile) {
  Graph graph{"config-actual-profile"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto attached = set_params(graph, source, {bind(op, "/overlay")});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  CompileOptions standard;
  const ExecutionPlan standard_plan = compile(graph, std::move(standard));
  ASSERT_TRUE(standard_plan.ok()) << standard_plan.json();

  CompileOptions research;
  research.profile = Profile::kResearch;
  const ExecutionPlan research_plan = compile(graph, std::move(research));
  EXPECT_TRUE(research_plan.ok()) << research_plan.json();
}

// REQ: COMPILE-072
// REQ: PARAM-037
// REQ: PARAM-039
// Graph: graph["config-actual-deployment"]
//   └─ partition["worker"]
//      └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, ConfigInputUsesTheActualDeployment) {
  Graph graph{"config-actual-deployment"};
  auto op = graph.op<PartialCodecOperator>("op");
  graph.partition("worker").add(op);
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto attached = set_params(graph, source, {bind(op, "/overlay")});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  CompileOptions placed;
  placed.deployment.bind_thread_placement("worker", ThreadPlacement{});
  const ExecutionPlan placed_plan = compile(graph, std::move(placed));
  EXPECT_TRUE(placed_plan.ok()) << placed_plan.json();
}

// REQ: COMPILE-072
// REQ: PARAM-037
// REQ: PARAM-039
// Graph: graph["config-actual-profile-rejection"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, ActualProfileStillRejectsInvalidConfigInput) {
  Graph graph{"config-actual-profile-rejection"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto attached = set_params(graph, source, {bind(op, "/overlay")});
  ASSERT_TRUE(attached.has_value()) << attached.error().token();

  CompileOptions regulated;
  regulated.profile = Profile::kRegulated;
  const ExecutionPlan regulated_plan = compile(graph, std::move(regulated));
  EXPECT_FALSE(regulated_plan.ok());
  EXPECT_TRUE(has_diagnostic(regulated_plan, "PARAMETER_SCHEMA_INCOMPLETE"));
}

// REQ: PARAM-034
// REQ: PARAM-037
// REQ: PARAM-039
// Graph: graph["config-field-failures"]
//   └─ op<PartialCodecOperator>("op")
// Two failure sub-cases: unknown field in load-time decode, and unauthorized field
// (external_access=kNone) in runtime decode.
TEST(ParameterConfigProvenance, UnknownAndUnauthorizedFieldsFailClosed) {
  // Key setup: unknown sub-case provides field name "unknown" not in the partial schema;
  // unauthorized sub-case uses runtime_catalog(kNone) making the field non-adapter-eligible.
  Graph graph{"config-field-failures"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document unknown = parse_json(R"({"overlay":{"unknown":1}})");
  auto unknown_result = set_params(graph, unknown, {bind(op, "/overlay")});
  ASSERT_FALSE(unknown_result.has_value());
  EXPECT_EQ(unknown_result.error().token(), "PARAM_CONFIG_FIELD_UNKNOWN");

  const config::Document runtime = parse_json(R"({"values":{"managed":"configured"}})");
  auto operator_path = OperatorPath::parse("op");
  ASSERT_TRUE(operator_path.has_value());
  const RuntimeParamConfigBinding binding{.operator_path = std::move(*operator_path),
                                          .values_path = path("/values")};
  auto unauthorized =
      decode_runtime_patches(runtime, runtime_catalog(ParamExternalAccess::kNone), binding);
  ASSERT_FALSE(unauthorized.has_value());
  EXPECT_EQ(unauthorized.error().token(), "PARAM_CONFIG_FIELD_NOT_ADAPTER_ELIGIBLE");
}

}  // namespace
}  // namespace holoscan::params::test
