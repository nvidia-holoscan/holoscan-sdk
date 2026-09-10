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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/config/codec.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/params/config.hpp>

#include "compiler/execution_plan_state.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "core/parameter_catalog_internal.hpp"
#include "core/parameter_override_set_internal.hpp"
#include "core/parameter_rebind_support.hpp"

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
    return ConfigValue::object(entries);
  }
};

namespace holoscan::params::test {
namespace {

// Test DSL helpers: parse_json wraps config_io::parse with an assertion and a null fallback;
// path wraps config::Path::parse similarly.
[[nodiscard]] config::Document parse_json(std::string_view source) {
  auto parsed = config_io::parse(source, config_io::Format::kJson);
  EXPECT_TRUE(parsed.has_value());
  if (parsed) {
    return std::move(*parsed);
  }
  return config::Document::from_value(config::ConfigValue::null()).value();
}

[[nodiscard]] config::Path path(std::string_view value) {
  auto parsed = config::Path::parse(value);
  EXPECT_TRUE(parsed.has_value());
  return parsed ? std::move(*parsed) : config::Path::root();
}

// Returns true if any diagnostic in the plan carries the given token string.
[[nodiscard]] bool has_diagnostic(const ExecutionPlan& plan, std::string_view token) {
  return std::ranges::any_of(plan.diagnostics(),
                             [&](const Diagnostic& item) { return item.token == token; });
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
// Graph: graph["partial-config-binding"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, PartialSchemaOverlayPreservesTypedUnmanagedBaseline) {
  // Key setup: baseline has managed="typed-managed", local_only="keep-local"; JSON document
  // provides only managed="configured". The partial codec must keep local_only from the baseline.
  Graph graph{"partial-config-binding"};
  auto op = graph.op<PartialCodecOperator>("op");
  op.set_params(PartialCodecParams{.managed = "typed-managed", .local_only = "keep-local"});
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");

  auto decoded = decode_overrides(source, {bind(op, "/overlay")});

  // Key assertion: managed field is patched; local_only is preserved; provenance records
  // the exact source path, semantic digest, and raw digest of the config document.
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
  const auto& private_state = holoscan::detail::ParamOverrideSetAccess::state(*decoded);
  ASSERT_EQ(private_state.operators.size(), 1U);
  const auto* value =
      static_cast<const PartialCodecParams*>(private_state.operators.front().resolved_value.get());
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(value->managed, "configured");
  EXPECT_EQ(value->local_only, "keep-local");

  ASSERT_EQ(decoded->provenance().config_fields.size(), 1U);
  EXPECT_EQ(decoded->provenance().context_binding, ParamOverrideContextBinding::kCompile);
  EXPECT_EQ(decoded->provenance().admission_context_digest, ParamSchemaHash{});
  const auto& source_field = decoded->provenance().config_fields.front();
  const auto source_digest = source.digest();
  EXPECT_EQ(source_field.document.semantic_digest, source_digest.semantic.sha256);
  ASSERT_TRUE(source_digest.source.has_value());
  const auto raw_source_digest = source_digest.source.value_or(config::RawSourceDigest{});
  EXPECT_TRUE(source_field.document.has_raw_source);
  EXPECT_EQ(source_field.document.raw_digest, raw_source_digest.sha256);
  EXPECT_EQ(source_field.source_path, "/overlay/managed");
  EXPECT_EQ(source_field.source_ordinal, 0U);
  EXPECT_EQ(source_field.target_operator_path, "op");
  EXPECT_EQ(source_field.target_schema_id, "holoscan.test.PartialCodecParams");

  CompileOptions options;
  options.parameter_overrides = *decoded;
  const ExecutionPlan plan = compile(graph, std::move(options));
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
}

// REQ: PARAM-034
// REQ: PARAM-037
// Graph: graph["config-source-plan-generation"]
//   └─ op<PartialCodecOperator>("op")
// Compact and spaced JSON documents encode the same value but have different raw bytes,
// so they must produce different plan generations even with identical semantic content.
TEST(ParameterConfigProvenance, ExactRawSourceIdentityChangesPlanGeneration) {
  // Key setup: compact vs spaced JSON → same semantic digest, different raw_source digest.
  Graph graph{"config-source-plan-generation"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document compact = parse_json(R"({"overlay":{"managed":"configured"}})");
  const config::Document spaced =
      parse_json("{ \"overlay\" : { \"managed\" : \"configured\" } }\n");
  const auto compact_digest = compact.digest();
  const auto spaced_digest = spaced.digest();
  ASSERT_EQ(compact_digest.semantic.sha256, spaced_digest.semantic.sha256);
  ASSERT_TRUE(compact_digest.source.has_value());
  ASSERT_TRUE(spaced_digest.source.has_value());
  const auto compact_source = compact_digest.source.value_or(config::RawSourceDigest{});
  const auto spaced_source = spaced_digest.source.value_or(config::RawSourceDigest{});
  ASSERT_NE(compact_source.sha256, spaced_source.sha256);

  auto first = decode_overrides(compact, {bind(op, "/overlay")});
  auto second = decode_overrides(spaced, {bind(op, "/overlay")});
  ASSERT_TRUE(first.has_value()) << first.error().token();
  ASSERT_TRUE(second.has_value()) << second.error().token();
  CompileOptions first_options;
  first_options.parameter_overrides = *first;
  CompileOptions second_options;
  second_options.parameter_overrides = *second;
  const ExecutionPlan first_plan = compile(graph, std::move(first_options));
  const ExecutionPlan second_plan = compile(graph, std::move(second_options));
  ASSERT_TRUE(first_plan.ok()) << first_plan.json();
  ASSERT_TRUE(second_plan.ok()) << second_plan.json();
  const auto* first_state = holoscan::detail::ExecutionPlanAccess::state(first_plan);
  const auto* second_state = holoscan::detail::ExecutionPlanAccess::state(second_plan);
  ASSERT_NE(first_state, nullptr);
  ASSERT_NE(second_state, nullptr);
  // Key assertion: plan_generation must differ when the raw source bytes differ.
  EXPECT_NE(first_state->plan_generation, second_state->plan_generation);
}

// REQ: PARAM-034
// REQ: PARAM-037
// Graph: graph["config-provenance-revalidation"]
//   └─ op<PartialCodecOperator>("op")
// A bit-flipped semantic digest must fail before any values are applied.
TEST(ParameterConfigProvenance, CompilerRejectsStaleOverrideDigest) {
  // Key setup: stale case manually XORs the first byte of the override set's semantic_digest.
  Graph graph{"config-provenance-revalidation"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto stale = decode_overrides(source, {bind(op, "/overlay")});
  ASSERT_TRUE(stale.has_value()) << stale.error().token();
  auto& stale_state = const_cast<holoscan::detail::ParamOverrideSetState&>(
      holoscan::detail::ParamOverrideSetAccess::state(*stale));
  stale_state.semantic_digest.value.front() ^= std::byte{1U};
  CompileOptions stale_options;
  stale_options.parameter_overrides = *stale;
  const ExecutionPlan stale_plan = compile(graph, std::move(stale_options));
  // Key assertion: stale digest -> RECONSTRUCTION_UNAVAILABLE.
  EXPECT_FALSE(stale_plan.ok());
  EXPECT_TRUE(has_diagnostic(stale_plan, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE"));
}

// REQ: COMPILE-072
// REQ: PARAM-037
// Graph: graph["config-invalid-context-binding"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, CompilerRejectsInvalidConfigContextBindingState) {
  Graph graph{"config-invalid-context-binding"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  const auto captured = holoscan::detail::capture_parameter_rebind_graph(graph);
  ASSERT_TRUE(captured.has_value()) << captured.error().message;
  const ParamSchemaHash actual_context = captured->admission_context_digest;
  ASSERT_NE(actual_context, ParamSchemaHash{});
  const auto expect_rejected = [&](const char* case_name, const auto& mutate) {
    SCOPED_TRACE(case_name);
    auto decoded = decode_overrides(source, {bind(op, "/overlay")});
    ASSERT_TRUE(decoded.has_value()) << decoded.error().token();
    auto& state = const_cast<holoscan::detail::ParamOverrideSetState&>(
        holoscan::detail::ParamOverrideSetAccess::state(*decoded));
    mutate(state);
    state.semantic_digest =
        holoscan::detail::ParamOverrideSetAccess::recompute_semantic_digest(*decoded);
    ASSERT_EQ(state.semantic_digest,
              holoscan::detail::ParamOverrideSetAccess::recompute_semantic_digest(*decoded));

    CompileOptions options;
    options.parameter_overrides = *decoded;
    const ExecutionPlan plan = compile(graph, std::move(options));
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(has_diagnostic(plan, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE"));
  };

  expect_rejected("unknown binding", [](auto& state) {
    const auto unknown = static_cast<ParamOverrideContextBinding>(255U);
    state.context_binding = unknown;
    state.provenance.context_binding = unknown;
  });
  expect_rejected("state and provenance disagree", [](auto& state) {
    state.provenance.context_binding = ParamOverrideContextBinding::kExact;
  });
  expect_rejected("compile-context digest is nonempty", [](auto& state) {
    state.admission_context_digest.value.front() = std::byte{1U};
    state.provenance.admission_context_digest = state.admission_context_digest;
  });
  expect_rejected("exact-context digest is empty", [](auto& state) {
    state.context_binding = ParamOverrideContextBinding::kExact;
    state.provenance.context_binding = ParamOverrideContextBinding::kExact;
  });
  expect_rejected("exact-context retains configuration provenance", [actual_context](auto& state) {
    ASSERT_FALSE(state.provenance.config_fields.empty());
    state.context_binding = ParamOverrideContextBinding::kExact;
    state.provenance.context_binding = ParamOverrideContextBinding::kExact;
    state.admission_context_digest = actual_context;
    state.provenance.admission_context_digest = actual_context;
  });
}

// REQ: COMPILE-072
// REQ: PARAM-037
// Graph: graph["config-unprovenanced-contribution"]
//   └─ op<ConfigProvenancePairOperator>("op")
TEST(ParameterConfigProvenance, CompilerRejectsUnprovenancedContributionInCompileContextOverride) {
  Graph graph{"config-unprovenanced-contribution"};
  auto op = graph.op<ConfigProvenancePairOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"primary":3}})");
  auto decoded = decode_overrides(source, {bind(op, "/overlay")});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();

  auto& state = const_cast<holoscan::detail::ParamOverrideSetState&>(
      holoscan::detail::ParamOverrideSetAccess::state(*decoded));
  ASSERT_EQ(state.operators.size(), 1U);
  ASSERT_EQ(state.provenance.config_fields.size(), 1U);
  auto& override_record = state.operators.front();
  ASSERT_EQ(override_record.contributions.size(), 1U);
  ASSERT_EQ(override_record.contributions.front().config_provenance_index,
            std::optional<std::uint32_t>{0U});
  EXPECT_EQ(override_record.contributions.front().field,
            state.provenance.config_fields.front().target_field_id);
  EXPECT_EQ(override_record.contributions.front().full_id,
            state.provenance.config_fields.front().target_full_id);
  const auto* resolved =
      static_cast<const ConfigProvenancePairParams*>(override_record.resolved_value.get());
  ASSERT_NE(resolved, nullptr);
  ConfigProvenancePairParams malformed = *resolved;
  malformed.secondary = 99U;
  override_record.resolved_value = std::make_shared<const ConfigProvenancePairParams>(malformed);

  const ParamSchemaDescriptor* const schema = param_schema_descriptor<ConfigProvenancePairParams>();
  ASSERT_NE(schema, nullptr);
  const auto secondary = std::ranges::find_if(
      schema->fields,
      [](const ParamFieldDescriptor& field) { return field.canonical_name == "secondary"; });
  ASSERT_NE(secondary, schema->fields.end());
  const auto resolved_digest = holoscan::detail::parameter_value_digest(
      param_type_descriptor<ConfigProvenancePairParams>(), schema, &malformed, false);
  ASSERT_TRUE(resolved_digest.available());
  override_record.resolved_value_digest = resolved_digest.digest;
  override_record.contributions.push_back(holoscan::detail::ParamOverrideFieldOriginRecord{
      .field = secondary->id,
      .full_id = secondary->full_id,
      .origin = holoscan::detail::ParamOverrideContributionOrigin::kCaptured,
  });
  state.semantic_digest =
      holoscan::detail::ParamOverrideSetAccess::recompute_semantic_digest(*decoded);
  ASSERT_EQ(state.semantic_digest,
            holoscan::detail::ParamOverrideSetAccess::recompute_semantic_digest(*decoded));

  CompileOptions options;
  options.parameter_overrides = *decoded;
  const ExecutionPlan plan = compile(graph, std::move(options));
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "PARAMETER_OVERRIDE_PROVENANCE_MISMATCH")) << plan.json();
  const auto* plan_state = holoscan::detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(plan_state, nullptr);
  ASSERT_EQ(plan_state->executable_factories.operators.size(), 1U);
  const auto* retained = static_cast<const ConfigProvenancePairParams*>(
      plan_state->executable_factories.operators.front().factory->initial_parameter_value());
  ASSERT_NE(retained, nullptr);
  EXPECT_EQ(retained->primary, 1U);
  EXPECT_EQ(retained->secondary, 2U);
}

// REQ: COMPILE-072
// REQ: PARAM-037
// Graph: graph["config-rebind-evidence"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, CompilerRejectsRebindEvidenceInCompileContextOverride) {
  Graph graph{"config-rebind-evidence"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto decoded = decode_overrides(source, {bind(op, "/overlay")});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();

  auto& state = const_cast<holoscan::detail::ParamOverrideSetState&>(
      holoscan::detail::ParamOverrideSetAccess::state(*decoded));
  state.provenance.migrations.push_back(ParamRebindMigrationProvenance{
      .from_schema_id = "source",
      .from_schema_version = 1U,
      .to_schema_id = "target",
      .to_schema_version = 2U,
  });
  state.semantic_digest =
      holoscan::detail::ParamOverrideSetAccess::recompute_semantic_digest(*decoded);
  ASSERT_EQ(state.semantic_digest,
            holoscan::detail::ParamOverrideSetAccess::recompute_semantic_digest(*decoded));

  CompileOptions options;
  options.parameter_overrides = *decoded;
  const ExecutionPlan plan = compile(graph, std::move(options));
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(has_diagnostic(plan, "PARAMETER_OVERRIDE_RECONSTRUCTION_UNAVAILABLE"));
}

// REQ: COMPILE-072
// REQ: PARAM-037
// Graph: graph["config-actual-profile"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, ConfigOverrideUsesTheActualValidCompileProfile) {
  Graph graph{"config-actual-profile"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto decoded = decode_overrides(source, {bind(op, "/overlay")});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();

  CompileOptions standard;
  standard.parameter_overrides = *decoded;
  const ExecutionPlan standard_plan = compile(graph, std::move(standard));
  ASSERT_TRUE(standard_plan.ok()) << standard_plan.json();

  CompileOptions research;
  research.profile = Profile::kResearch;
  research.parameter_overrides = *decoded;
  const ExecutionPlan research_plan = compile(graph, std::move(research));
  EXPECT_TRUE(research_plan.ok()) << research_plan.json();
}

// REQ: COMPILE-072
// REQ: PARAM-037
// Graph: graph["config-actual-deployment"]
//   └─ partition["worker"]
//      └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, ConfigOverrideUsesTheActualDeployment) {
  Graph graph{"config-actual-deployment"};
  auto op = graph.op<PartialCodecOperator>("op");
  graph.partition("worker").add(op);
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto decoded = decode_overrides(source, {bind(op, "/overlay")});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();

  CompileOptions placed;
  placed.deployment.bind_thread_placement("worker", ThreadPlacement{});
  placed.parameter_overrides = *decoded;
  const ExecutionPlan placed_plan = compile(graph, std::move(placed));
  EXPECT_TRUE(placed_plan.ok()) << placed_plan.json();
  EXPECT_FALSE(has_diagnostic(placed_plan, "PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH"));
}

// REQ: COMPILE-072
// REQ: PARAM-037
// Graph: graph["config-actual-profile-rejection"]
//   └─ op<PartialCodecOperator>("op")
TEST(ParameterConfigProvenance, ActualProfileStillRejectsInvalidConfigOverride) {
  Graph graph{"config-actual-profile-rejection"};
  auto op = graph.op<PartialCodecOperator>("op");
  const config::Document source = parse_json(R"({"overlay":{"managed":"configured"}})");
  auto decoded = decode_overrides(source, {bind(op, "/overlay")});
  ASSERT_TRUE(decoded.has_value()) << decoded.error().token();

  CompileOptions regulated;
  regulated.profile = Profile::kRegulated;
  regulated.parameter_overrides = *decoded;
  const ExecutionPlan regulated_plan = compile(graph, std::move(regulated));
  EXPECT_FALSE(regulated_plan.ok());
  EXPECT_TRUE(has_diagnostic(regulated_plan, "PARAMETER_SCHEMA_INCOMPLETE"));
  EXPECT_FALSE(has_diagnostic(regulated_plan, "PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH"));
}

// REQ: PARAM-037
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
  auto unknown_result = decode_overrides(unknown, {bind(op, "/overlay")});
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
