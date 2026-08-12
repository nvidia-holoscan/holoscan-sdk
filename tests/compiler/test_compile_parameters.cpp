/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/compile_platform.hpp"
#include "compiler/deployment_plan_lowering.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/parameter_generation_zero.hpp"
#include "compiler/parameter_plan_lowering.hpp"
#include "compiler/profile_admission.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/parameter_override_set_internal.hpp"
#include "core/parameter_rebind_support.hpp"
#include "core/parameter_schema_test_access.hpp"

// GTest fatal assertions establish optional preconditions that clang-tidy does not model.
// NOLINTBEGIN(bugprone-unchecked-optional-access)
namespace holoscan {

// Full-catalog params with diverse field policies: gain (proven, runtime-mutable,
// ranged, external-tuning), secret (sensitive), internal_counter (internal).
// Representative test: BuildsDeterministicPolicyFilteredCatalogAndFiniteCapacity.
struct CompileCatalogParams {
  std::uint32_t gain{1U};
  std::uint32_t secret{7U};
  std::uint32_t internal_counter{9U};
};

HOLOSCAN_PARAM_SCHEMA(
    CompileCatalogParams,
    holoscan::detail::ParamSchemaTestAccess::proven(
        holoscan::param_schema("nvidia.holoscan.test.CompileCatalogParams", 1)),
    holoscan::field<&CompileCatalogParams::gain>("gain")
        .runtime_mutable()
        .range(0U, 8U)
        .persisted()
        .external_tuning(),
    holoscan::field<&CompileCatalogParams::secret>("secret").sensitive(),
    holoscan::field<&CompileCatalogParams::internal_counter>("internal_counter").internal());

// Partial schema: 'declared' is published to the catalog; 'local_only' is absent.
// Representative test: AppliesOpaquePartialCompleteAndProofAdmissionMatrix.
struct CompilePartialParams {
  std::uint32_t declared{1U};
  std::uint32_t local_only{2U};
};

HOLOSCAN_PARAM_SCHEMA(
    CompilePartialParams,
    holoscan::param_schema("nvidia.holoscan.test.CompilePartialParams", 1).partial(),
    holoscan::field<&CompilePartialParams::declared>("declared"));

// Params struct with no HOLOSCAN_PARAM_SCHEMA: the compiler treats it as opaque
// (kOpaqueLocal). Used in RejectsFieldPatchAgainstOpaqueTypedAggregate and the
// opaque branch of AppliesOpaquePartialCompleteAndProofAdmissionMatrix.
struct CompileOpaqueParams {
  std::uint32_t local_only{1U};
};

// Complete schema without the proven() mark: disposition is kCompleteAuthorAsserted.
// Strict mode rejects it with PARAMETER_SCHEMA_INCOMPLETE.
// Representative test: AppliesOpaquePartialCompleteAndProofAdmissionMatrix.
struct CompileAuthorAssertedParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileAuthorAssertedParams,
                      holoscan::param_schema("nvidia.holoscan.test.CompileAuthorAssertedParams", 1),
                      holoscan::field<&CompileAuthorAssertedParams::value>("value"));

// UTF-8 string field (max 32 bytes): dynamic heap storage forces kColdOwnedTyped
// materialization and blocks bounded and inline-only storage modes.
// Representative test: MakesDynamicStringVectorAndPathMaterializationExplicit.
struct CompileDynamicParams {
  std::string label{"camera"};
};

HOLOSCAN_PARAM_SCHEMA(CompileDynamicParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("nvidia.holoscan.test.CompileDynamicParams", 1)),
                      holoscan::field<&CompileDynamicParams::label>("label")
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(32U));

// Vector field with a bounded element count: owns heap storage → kColdOwnedTyped.
// Representative test: MakesDynamicStringVectorAndPathMaterializationExplicit.
struct CompileVectorParams {
  std::vector<std::uint32_t> values{1U, 2U};
};

HOLOSCAN_PARAM_SCHEMA(CompileVectorParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("nvidia.holoscan.test.CompileVectorParams", 1)),
                      holoscan::field<&CompileVectorParams::values>("values").max_elements(8U));

// std::filesystem::path field: always cold-owned regardless of storage mode.
// Representative test: MakesDynamicStringVectorAndPathMaterializationExplicit.
struct CompilePathParams {
  std::filesystem::path value{"model.plan"};
};

HOLOSCAN_PARAM_SCHEMA(CompilePathParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("nvidia.holoscan.test.CompilePathParams", 1)),
                      holoscan::field<&CompilePathParams::value>("value"));

// Proven schema where persistence, access, and sensitivity are left at their inferred
// defaults; fails strict-mode compilation with PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN.
// Representative test: RequiresExplicitFieldPolicyEvidenceWhenRequestedOrRegulated.
struct CompileInferredPolicyParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileInferredPolicyParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.CompileInferredPolicyParams", 1)),
                      holoscan::field<&CompileInferredPolicyParams::value>("value"));

// Proven schema with every field declaring persistence, access, and sensitivity
// explicitly; passes strict-mode and regulated-profile compilation.
// Representative test: RequiresExplicitFieldPolicyEvidenceWhenRequestedOrRegulated.
struct CompileExplicitPolicyParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileExplicitPolicyParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.CompileExplicitPolicyParams", 1)),
                      holoscan::field<&CompileExplicitPolicyParams::value>("value")
                          .persistence(ParamPersistence::kPersisted)
                          .external_access(ParamExternalAccess::kNone)
                          .sensitivity(ParamSensitivity::kNormal));

// Proven schema with runtime_mutable + fully explicit field policies: the minimal
// evidence that admits active mutation on Standard and Research profiles.
// Representative test: AdmitsActiveMutationOnlyForStandardAndResearchProfiles.
struct CompileExplicitRuntimeParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileExplicitRuntimeParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(holoscan::param_schema(
                          "nvidia.holoscan.test.CompileExplicitRuntimeParams", 1)),
                      holoscan::field<&CompileExplicitRuntimeParams::value>("value")
                          .runtime_mutable()
                          .persistence(ParamPersistence::kPersisted)
                          .external_access(ParamExternalAccess::kTuning)
                          .sensitivity(ParamSensitivity::kNormal));

// Multi-field params for CompileReplayOperator: setup_shape controls the input port
// name in setup(), contract_valid selects the contract trigger, gain is a tunable
// field. Used in ReplaysResolved*, RejectsParameterDependent*, and
// RejectsInvalidContract* tests.
struct CompileReplayParams {
  std::uint32_t setup_shape{1U};
  bool contract_valid{true};
  std::uint32_t gain{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileReplayParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("nvidia.holoscan.test.CompileReplayParams", 1)),
                      holoscan::field<&CompileReplayParams::setup_shape>("setup_shape"),
                      holoscan::field<&CompileReplayParams::contract_valid>("contract_valid"),
                      holoscan::field<&CompileReplayParams::gain>("gain"));

// Single gain field admitted for lifecycle-stage mutation (kConfigure).
// Representative test: LifecycleOptInAdmitsPrivateCandidateWithoutComputeIngress.
struct CompileLifecycleParams {
  std::uint32_t gain{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileLifecycleParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("nvidia.holoscan.test.CompileLifecycleParams", 1)),
                      holoscan::field<&CompileLifecycleParams::gain>("gain").lifecycle_mutable(
                          LifecycleStage::kConfigure));

namespace {

// Minimal single-operator template; compute() is a no-op so tests focus on schema,
// policy, and override compilation without runtime side effects. Used by most
// ParameterCompile tests (e.g. BuildsDeterministicPolicyFilteredCatalogAndFiniteCapacity,
// AdmitsActiveMutationOnlyForStandardAndResearchProfiles).
template <typename ParamsT>
class CompileParameterOperator final : public Operator<ParamsT> {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator whose port declaration and trigger contract both depend on runtime params:
// setup() names the input port based on setup_shape; contract() triggers on 'input'
// or the undeclared member based on contract_valid. Used in replay parameter tests.
class CompileReplayOperator final : public Operator<CompileReplayParams> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, params().setup_shape == 1U ? "input" : "changed-input").queue_depth(2U);
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(params().contract_valid ? OnEach{input} : OnEach{undeclared});
    return result;
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Input<std::uint32_t> undeclared;
};

template <typename T>
concept HasValueMember = requires(const T& value) { value.value; };

template <typename T>
concept HasValueDigestMember = requires(const T& value) { value.value_digest; };

template <typename T>
concept HasValueOriginMember = requires(const T& value) { value.value_origin; };

static_assert(!HasValueMember<ParamFieldCatalogRecord>);
static_assert(!HasValueDigestMember<ParamFieldCatalogRecord>);
static_assert(!HasValueOriginMember<ParamFieldCatalogRecord>);

// REQ: COMPILE-019
TEST(ParameterCompile, ClassifiesInjectedFieldAndInstanceProjectionCollisions) {
  ParamSchemaHash first_full{};
  first_full.value[0] = std::byte{0x11U};
  ParamSchemaHash second_full{};
  second_full.value[0] = std::byte{0x22U};

  const std::vector<detail::ParameterFieldProjectionIdentity> fields{{17U, first_full}};
  const auto field_collision =
      detail::classify_parameter_field_projection(fields, 17U, second_full);
  EXPECT_EQ(field_collision, detail::ParameterProjectionCollision::kFieldId);
  EXPECT_EQ(detail::parameter_projection_collision_token(field_collision),
            "PARAMETER_FIELD_ID_COLLISION");
  EXPECT_EQ(detail::classify_parameter_field_projection(fields, 18U, second_full),
            detail::ParameterProjectionCollision::kNone);

  const std::vector<detail::ParameterInstanceProjectionIdentity> instances{{23U, first_full}};
  const auto compact_collision =
      detail::classify_parameter_instance_projection(instances, 23U, second_full);
  EXPECT_EQ(compact_collision, detail::ParameterProjectionCollision::kInstanceId);
  EXPECT_EQ(detail::parameter_projection_collision_token(compact_collision),
            "PARAMETER_INSTANCE_ID_COLLISION");
  EXPECT_EQ(detail::classify_parameter_instance_projection(instances, 23U, first_full),
            detail::ParameterProjectionCollision::kFieldId);
  EXPECT_EQ(detail::classify_parameter_instance_projection(instances, 0U, second_full),
            detail::ParameterProjectionCollision::kInstanceId);
}

// Looks up a named field from the CompileCatalogParams schema; fails the test if absent.
[[nodiscard]] const ParamFieldDescriptor& catalog_field(std::string_view name) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<CompileCatalogParams>();
  const auto found = std::find_if(schema->fields.begin(),
                                  schema->fields.end(),
                                  [&](const auto& item) { return item.canonical_name == name; });
  EXPECT_NE(found, schema->fields.end());
  return *found;
}

// Returns CompileOptions with runtime updates enabled and preallocated-bounded storage.
[[nodiscard]] CompileOptions runtime_parameter_options() {
  CompileOptions options;
  options.parameters.runtime_updates = ParamRuntimeUpdatePolicy::kRequirePreallocated;
  options.parameters.storage = ParamStorageMode::kPreallocatedBounded;
  return options;
}

// Returns a sealed RuntimePackageCapabilityManifest carrying the evidence tokens
// required for RealTime or Regulated profile admission.
[[nodiscard]] detail::RuntimePackageCapabilityManifest package_with_profile_evidence(
    Profile profile) {
  auto package = detail::runtime_package_capability_manifest();
  if (profile == Profile::kRealTime) {
    package.offers.push_back(detail::RuntimePackageCapabilityOffer{
        .id = "holoscan.profile.real_time.bounded_hot_path",
        .version = 1U,
        .evidence_id = "holoscan.profile.real_time.bounded_hot_path/v1",
    });
  }
  if (profile == Profile::kRegulated) {
    const auto offer =
        std::find_if(package.offers.begin(), package.offers.end(), [](const auto& item) {
          return item.id == "holoscan.profile.regulated.complete_evidence" && item.version == 1U;
        });
    if (offer != package.offers.end()) {
      offer->evidence_id = "holoscan.profile.regulated.complete_evidence/v1";
    }
  }
  EXPECT_TRUE(detail::seal_runtime_package_capability_manifest(package));
  return package;
}

// Compiles a single CompileParameterOperator<ParamsT> graph with profile-capability
// evidence injected into the runtime package; used for profiles that require proof.
template <typename ParamsT>
[[nodiscard]] ExecutionPlan compile_parameter_case_with_profile_evidence(
    std::string_view graph_name, CompileOptions options) {
  Graph graph{graph_name};
  (void)graph.op<CompileParameterOperator<ParamsT>>("operator");
  auto package = package_with_profile_evidence(options.profile);
  const auto view = detail::validate_runtime_package_capability_manifest(package);
  EXPECT_TRUE(view.has_value());
  if (!view) {
    return compile(graph, std::move(options));
  }
  return detail::compile_with_runtime_evidence(
      graph,
      std::move(options),
      detail::RuntimeCompileEvidenceView{.package_capabilities = view.value()});
}

// Returns generous LifecycleTransactionOptions so capacity is never the binding
// constraint in tests that focus on lifecycle parameter admission logic.
[[nodiscard]] LifecycleTransactionOptions ample_parameter_lifecycle_capacity() {
  return LifecycleTransactionOptions{
      .max_pending_requests = 64U,
      .max_pending_transactions = 64U,
      .max_receipts = 128U,
      .max_fenced_operators = 64U,
      .max_candidate_generations = 64U,
      .max_events = 1024U,
      .max_audit_records = 128U,
      .max_quiesce_records = 64U,
      .max_rollback_records = 64U,
  };
}

// Compiles a single CompileParameterOperator<ParamsT> named "operator"; graph_name
// is used only for tracing and uniqueness in multi-case tests.
template <typename ParamsT>
[[nodiscard]] ExecutionPlan compile_parameter_case(std::string_view graph_name,
                                                   CompileOptions options = {}) {
  Graph graph{graph_name};
  (void)graph.op<CompileParameterOperator<ParamsT>>("operator");
  return compile(graph, std::move(options));
}

// Computes the admission context digest that ties an override to a specific
// (profile × deployment × migration-catalog) tuple. Overrides presented under a
// different context are rejected with PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH.
[[nodiscard]] ParamSchemaHash parameter_admission_context(
    Graph& graph, const CompileOptions& options,
    ParamSchemaHash migration_catalog = detail::empty_parameter_migration_catalog_digest()) {
  if (options.profile == Profile::kStandard && options.deployment.thread_placements().empty() &&
      options.deployment.generated_endpoint_names().empty() &&
      options.deployment.tensor_output_devices().empty() &&
      migration_catalog == detail::empty_parameter_migration_catalog_digest()) {
    const auto captured = detail::capture_parameter_rebind_graph(graph);
    if (!captured) {
      throw std::logic_error{"parameter admission-context fixture capture failed"};
    }
    return captured->admission_context_digest;
  }
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  auto profile = detail::resolve_profile_admission(*snapshot, options.profile);
  profile.manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
  profile.manifest.target_triple = std::string(detail::compile_target_triple());
  profile.manifest.producer_build_id = "holoscan-core-5.0";
  CompileOptions baseline_options;
  baseline_options.profile = options.profile;
  baseline_options.deployment = options.deployment;
  const auto baseline = compile(graph, std::move(baseline_options));
  const auto* baseline_state = detail::ExecutionPlanAccess::state(baseline);
  if (baseline_state == nullptr) {
    throw std::logic_error{"parameter admission-context fixture compile failed"};
  }
  auto partitions = baseline_state->execution_partitions;
  const auto deployment = detail::lower_deployment_plan(
      *snapshot, options.deployment, profile.effective_profiles(), partitions);
  return detail::parameter_admission_context_digest(
      profile.manifest, deployment, migration_catalog);
}

// Computes the parameter value digest for a typed instance; throws on failure,
// which indicates a test-fixture misconfiguration rather than a product bug.
template <typename ParamsT>
[[nodiscard]] ParamSchemaHash required_value_digest(const ParamsT& value) {
  const auto digest = detail::parameter_value_digest(
      param_type_descriptor<ParamsT>(), param_schema_descriptor<ParamsT>(), &value, false);
  if (!digest.available()) {
    throw std::logic_error{"invalid parameter digest test fixture"};
  }
  return digest.digest;
}

// Builds a complete ParamOverrideSet with one captured field change on the "target"
// operator; changed_field names the single contributed field for replay tests.
[[nodiscard]] ParamOverrideSet replay_override(Graph& graph, const CompileOptions& options,
                                               CompileReplayParams resolved,
                                               std::string_view changed_field) {
  const ParamSchemaDescriptor* const schema = param_schema_descriptor<CompileReplayParams>();
  const ParamTypeDescriptor& type = param_type_descriptor<CompileReplayParams>();
  const CompileReplayParams baseline{};
  const auto baseline_digest = detail::parameter_value_digest(type, schema, &baseline, false);
  const auto field =
      std::find_if(schema->fields.begin(), schema->fields.end(), [&](const auto& item) {
        return item.canonical_name == changed_field;
      });
  if (!baseline_digest.available() || field == schema->fields.end()) {
    throw std::logic_error{"invalid replay override test fixture"};
  }
  const ParamSchemaHash resolved_digest = required_value_digest(resolved);
  std::vector<detail::ParamOverrideOperatorState> operators;
  operators.push_back(detail::ParamOverrideSetAccess::resolved_operator(
      "target",
      resolved,
      baseline_digest.digest,
      resolved_digest,
      {{.field = field->id,
        .full_id = field->full_id,
        .origin = detail::ParamOverrideContributionOrigin::kCaptured}}));
  return detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(), std::move(operators), parameter_admission_context(graph, options));
}

// Builds a captured ParamOverrideOperatorState for the 'gain' field of
// CompileCatalogParams; reused across override-validation and context-mismatch tests.
[[nodiscard]] detail::ParamOverrideOperatorState catalog_gain_override_record(
    std::uint32_t gain_value) {
  const CompileCatalogParams baseline{};
  CompileCatalogParams resolved = baseline;
  resolved.gain = gain_value;
  const ParamFieldDescriptor& gain = catalog_field("gain");
  return detail::ParamOverrideSetAccess::resolved_operator(
      "operator",
      resolved,
      required_value_digest(baseline),
      required_value_digest(resolved),
      {{.field = gain.id,
        .full_id = gain.full_id,
        .origin = detail::ParamOverrideContributionOrigin::kCaptured}});
}

// Wraps catalog_gain_override_record into a complete ParamOverrideSet bound to the
// given graph identity, admission context, and optional migration catalog hash.
[[nodiscard]] ParamOverrideSet catalog_gain_override(
    Graph& graph, const CompileOptions& context_options, std::uint32_t gain_value,
    ParamSchemaHash migration_catalog = detail::empty_parameter_migration_catalog_digest()) {
  std::vector<detail::ParamOverrideOperatorState> operators;
  operators.push_back(catalog_gain_override_record(gain_value));
  return detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(),
      std::move(operators),
      parameter_admission_context(graph, context_options, migration_catalog));
}

// REQ: COMPILE-019
// REQ: PARAM-003
// REQ: PARAM-009
TEST(ParameterCompile, BuildsDeterministicPolicyFilteredCatalogAndFiniteCapacity) {
  Graph graph{"parameter-catalog"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");

  // Key setup: two compilations under identical options to verify catalog reproducibility.
  const ExecutionPlan first = compile(graph, runtime_parameter_options());
  const ExecutionPlan second = compile(graph, runtime_parameter_options());
  ASSERT_TRUE(first.ok()) << first.json();
  ASSERT_TRUE(second.ok()) << second.json();
  ASSERT_EQ(first.operator_parameters().size(), 1U);
  EXPECT_EQ(first.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kCompleteProven);
  EXPECT_EQ(first.operator_parameters().front().materialization,
            ParameterMaterializationStrategy::kFixedObject);

  ASSERT_TRUE(first.parameter_runtime_capacity());
  EXPECT_TRUE(first.parameter_runtime_capacity()->runtime_updates_enabled);
  EXPECT_EQ(first.parameter_runtime_capacity()->enabled_operator_count, 1U);
  EXPECT_GE(first.parameter_runtime_capacity()->generation_slots_per_operator, 6U);
  EXPECT_GT(first.parameter_runtime_capacity()->parameter_arena_bytes, 0U);

  const ParameterCatalogView catalog = first.parameter_catalog();
  ASSERT_EQ(catalog.operators().size(), 1U);
  ASSERT_EQ(catalog.fields().size(), 2U);
  EXPECT_EQ(catalog.compatibility_digest(), second.parameter_catalog().compatibility_digest());
  EXPECT_TRUE(test::has_nonzero_byte(catalog.compatibility_digest().value));
  EXPECT_EQ(catalog.fields()[0].instance, second.parameter_catalog().fields()[0].instance);

  const detail::ExecutionPlanState* const first_state = detail::ExecutionPlanAccess::state(first);
  const detail::ExecutionPlanState* const second_state = detail::ExecutionPlanAccess::state(second);
  ASSERT_NE(first_state, nullptr);
  ASSERT_NE(second_state, nullptr);
  ASSERT_EQ(first_state->parameter_generation_zero_operators.size(), 1U);
  ASSERT_EQ(second_state->parameter_generation_zero_operators.size(), 1U);
  EXPECT_EQ(first_state->parameter_generation_zero_operators.front().baseline_origin,
            second_state->parameter_generation_zero_operators.front().baseline_origin);
  EXPECT_EQ(first_state->parameter_generation_zero_operators.front().value_digest.disposition,
            second_state->parameter_generation_zero_operators.front().value_digest.disposition);
  EXPECT_EQ(first_state->parameter_generation_zero_operators.front().value_digest.digest,
            second_state->parameter_generation_zero_operators.front().value_digest.digest);
  ASSERT_EQ(first_state->parameter_generation_zero_fields.size(),
            second_state->parameter_generation_zero_fields.size());
  for (std::size_t index = 0U; index < first_state->parameter_generation_zero_fields.size();
       ++index) {
    const auto& left = first_state->parameter_generation_zero_fields[index];
    const auto& right = second_state->parameter_generation_zero_fields[index];
    EXPECT_EQ(left.instance, right.instance);
    EXPECT_EQ(left.field, right.field);
    EXPECT_EQ(left.full_id, right.full_id);
    EXPECT_EQ(left.origin, right.origin);
    EXPECT_EQ(left.value_digest.disposition, right.value_digest.disposition);
    EXPECT_EQ(left.value_digest.digest, right.value_digest.digest);
  }

  // Key assertion: sensitive field is redacted; internal field is excluded from catalog.
  const auto sensitive =
      std::find_if(catalog.fields().begin(), catalog.fields().end(), [](const auto& field) {
        return field.sensitivity == ParamSensitivity::kSensitive;
      });
  ASSERT_NE(sensitive, catalog.fields().end());
  EXPECT_EQ(sensitive->projection, params::ParamProjectionStatus::kRedactedSensitive);
  EXPECT_TRUE(std::none_of(catalog.fields().begin(), catalog.fields().end(), [](const auto& field) {
    return field.visibility == ParamVisibility::kInternal;
  }));

  const auto operator_path = OperatorPath::parse("operator");
  const auto parameter_path = ParamPath::parse("gain");
  ASSERT_TRUE(operator_path);
  ASSERT_TRUE(parameter_path);
  const auto address = catalog.resolve(operator_path.value(), parameter_path.value());
  ASSERT_TRUE(address);
  const auto by_address = catalog.field(address.value());
  ASSERT_TRUE(by_address);
  const auto by_instance = catalog.field(by_address.value()->instance);
  ASSERT_TRUE(by_instance);
  EXPECT_EQ(by_address.value()->full_id, by_instance.value()->full_id);
  EXPECT_EQ(by_address.value()->update_disposition, ParameterFieldUpdateDisposition::kAdmitted);
}

// REQ: COMPILE-003
// REQ: COMPILE-019
// REQ: PARAM-017
// REQ: PARAM-018
TEST(ParameterCompile, AdmitsActiveMutationOnlyForStandardAndResearchProfiles) {
  for (const Profile profile : {Profile::kStandard, Profile::kResearch}) {
    CompileOptions options = runtime_parameter_options();
    options.profile = profile;
    const ExecutionPlan plan = compile_parameter_case<CompileExplicitRuntimeParams>(
        "parameter-active-profile-" + std::to_string(static_cast<int>(profile)), options);

    ASSERT_TRUE(plan.ok()) << plan.json();
    EXPECT_FALSE(test::has_diagnostic(plan, "PARAMETER_UPDATE_PROFILE_UNSUPPORTED"));
    const auto capacity = plan.parameter_runtime_capacity();
    ASSERT_TRUE(capacity.has_value());
    EXPECT_TRUE(capacity->runtime_updates_enabled);
    EXPECT_FALSE(capacity->lifecycle_updates_enabled);
    EXPECT_EQ(capacity->enabled_operator_count, 1U);
    EXPECT_GT(capacity->compute_ingress_capacity, 0U);
    EXPECT_GT(capacity->parameter_arena_bytes, 0U);
    ASSERT_EQ(plan.parameter_fields().size(), 1U);
    EXPECT_EQ(plan.parameter_fields().front().update_disposition,
              ParameterFieldUpdateDisposition::kAdmitted);
  }

  for (const Profile profile : {Profile::kRobotics, Profile::kRealTime, Profile::kRegulated}) {
    CompileOptions options = runtime_parameter_options();
    options.profile = profile;
    const ExecutionPlan plan =
        compile_parameter_case_with_profile_evidence<CompileExplicitRuntimeParams>(
            "parameter-frozen-profile-" + std::to_string(static_cast<int>(profile)), options);

    ASSERT_FALSE(plan.ok()) << plan.json();
    EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_UPDATE_PROFILE_UNSUPPORTED")) << plan.json();
    const auto capacity = plan.parameter_runtime_capacity();
    ASSERT_TRUE(capacity.has_value());
    EXPECT_FALSE(capacity->runtime_updates_enabled);
    EXPECT_FALSE(capacity->lifecycle_updates_enabled);
    EXPECT_EQ(capacity->enabled_operator_count, 0U);
    EXPECT_EQ(capacity->compute_ingress_capacity, 0U);
    EXPECT_EQ(capacity->parameter_arena_bytes, 0U);
    ASSERT_EQ(plan.parameter_fields().size(), 1U);
    EXPECT_EQ(plan.parameter_fields().front().update_disposition,
              ParameterFieldUpdateDisposition::kProfileDisallowed);
  }
}

// REQ: COMPILE-003
// REQ: COMPILE-019
// REQ: PARAM-017
TEST(ParameterCompile, KeepsGenerationZeroOnlyModeValidAcrossProfileMatrix) {
  for (const Profile profile : {Profile::kStandard,
                                Profile::kResearch,
                                Profile::kRobotics,
                                Profile::kRealTime,
                                Profile::kRegulated}) {
    CompileOptions options;
    options.profile = profile;
    // Key setup: nonzero generation_slots_per_operator must be silently discarded by
    // generation-zero plans rather than retained as dead storage in the capacity record.
    options.parameters.generation_slots_per_operator = 17U;
    const ExecutionPlan plan =
        compile_parameter_case_with_profile_evidence<CompileExplicitPolicyParams>(
            "parameter-generation-zero-profile-" + std::to_string(static_cast<int>(profile)),
            options);

    ASSERT_TRUE(plan.ok()) << plan.json();
    EXPECT_FALSE(test::has_diagnostic(plan, "PARAMETER_UPDATE_PROFILE_UNSUPPORTED"));
    const auto capacity = plan.parameter_runtime_capacity();
    ASSERT_TRUE(capacity.has_value());
    // Key assertion: all runtime capacity fields are zero; no storage allocated for updates.
    EXPECT_FALSE(capacity->runtime_updates_enabled);
    EXPECT_FALSE(capacity->lifecycle_updates_enabled);
    EXPECT_EQ(capacity->enabled_operator_count, 0U);
    EXPECT_EQ(capacity->max_pending_updates_per_operator, 0U);
    EXPECT_EQ(capacity->generation_slots_per_operator, 0U);
    EXPECT_EQ(capacity->compute_ingress_capacity, 0U);
    EXPECT_EQ(capacity->audit_capacity, 0U);
    EXPECT_EQ(capacity->event_capacity, 0U);
    EXPECT_EQ(capacity->max_event_cursors, 0U);
    EXPECT_EQ(capacity->parameter_arena_bytes, 0U);
    ASSERT_EQ(plan.operator_parameters().size(), 1U);
    EXPECT_EQ(plan.operator_parameters().front().initial_generation, 0U);
    ASSERT_EQ(plan.parameter_fields().size(), 1U);
    EXPECT_EQ(plan.parameter_fields().front().update_disposition,
              ParameterFieldUpdateDisposition::kCompileTime);
    const detail::ExecutionPlanState* const state = detail::ExecutionPlanAccess::state(plan);
    ASSERT_NE(state, nullptr);
    ASSERT_EQ(state->parameter_generation_zero_operators.size(), 1U);
    EXPECT_EQ(state->parameter_generation_zero_operators.front().owner,
              plan.operator_parameters().front().owner);
  }
}

// REQ: PARAM-021
TEST(ParameterPlan, LifecycleOptInAdmitsPrivateCandidateWithoutComputeIngress) {
  Graph graph{"parameter-lifecycle-private-candidate"};
  (void)graph.op<CompileParameterOperator<CompileLifecycleParams>>("operator");

  CompileOptions options;
  options.lifecycle_transactions = ample_parameter_lifecycle_capacity();
  const ExecutionPlan plan = compile(graph, options);

  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto capacity = plan.parameter_runtime_capacity();
  ASSERT_TRUE(capacity.has_value());
  EXPECT_FALSE(capacity->runtime_updates_enabled);
  EXPECT_TRUE(capacity->lifecycle_updates_enabled);
  EXPECT_EQ(capacity->compute_ingress_capacity, 0U);
  EXPECT_GE(capacity->generation_slots_per_operator,
            options.parameters.max_pending_updates_per_operator + 4U);
  ASSERT_EQ(plan.parameter_fields().size(), 1U);
  EXPECT_EQ(plan.parameter_fields().front().update_disposition,
            ParameterFieldUpdateDisposition::kAdmitted);

  ASSERT_EQ(plan.lifecycle_restart_targets().size(), 1U);
  EXPECT_EQ(plan.lifecycle_restart_targets().front().disposition,
            LifecycleRestartTargetDisposition::kAdmitted);
  EXPECT_TRUE(plan.lifecycle_restart_targets().front().parameter_origin_allowed);
  ASSERT_TRUE(plan.lifecycle_transaction_capacity().has_value());
  EXPECT_TRUE(plan.lifecycle_transaction_capacity()->parameter_origin_enabled);
}

// REQ: PARAM-021
TEST(ParameterPlan, LifecycleCandidateCapacityRejectsBeforeParameterAuthority) {
  Graph graph{"parameter-lifecycle-candidate-capacity"};
  (void)graph.op<CompileParameterOperator<CompileLifecycleParams>>("operator");

  CompileOptions options;
  // Key setup: max_candidate_generations=1 is below the minimum 2 required for lifecycle
  // parameter authority, triggering rejection before any field is admitted.
  options.parameters.max_pending_updates_per_operator = 1U;
  options.lifecycle_transactions = ample_parameter_lifecycle_capacity();
  options.lifecycle_transactions->max_candidate_generations = 1U;
  const ExecutionPlan plan = compile(graph, options);

  // Key assertion: insufficient candidate capacity blocks lifecycle updates entirely.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_LIFECYCLE_CAPACITY_INSUFFICIENT"));
  const auto capacity = plan.parameter_runtime_capacity();
  ASSERT_TRUE(capacity.has_value());
  EXPECT_FALSE(capacity->runtime_updates_enabled);
  EXPECT_FALSE(capacity->lifecycle_updates_enabled);
  EXPECT_EQ(capacity->compute_ingress_capacity, 0U);
  EXPECT_EQ(capacity->parameter_arena_bytes, 0U);
  EXPECT_GE(capacity->required_lifecycle_capacity.max_candidate_generations, 2U);
  ASSERT_EQ(plan.parameter_fields().size(), 1U);
  EXPECT_EQ(plan.parameter_fields().front().update_disposition,
            ParameterFieldUpdateDisposition::kCapacityUnavailable);

  ASSERT_TRUE(plan.lifecycle_transaction_capacity().has_value());
  EXPECT_FALSE(plan.lifecycle_transaction_capacity()->parameter_origin_enabled);
  EXPECT_TRUE(std::none_of(plan.lifecycle_restart_targets().begin(),
                           plan.lifecycle_restart_targets().end(),
                           [](const auto& target) {
                             return target.parameter_origin_allowed &&
                                    target.disposition ==
                                        LifecycleRestartTargetDisposition::kAdmitted;
                           }));
}

// REQ: COMPILE-019
// REQ: PARAM-007
// REQ: PARAM-009
TEST(ParameterCompile, AppliesCompleteOverrideAtomicallyAndRetainsProtectedOrigin) {
  Graph graph{"parameter-override"};
  auto op = graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  const CompileCatalogParams baseline{.gain = 2U, .secret = 7U, .internal_counter = 9U};
  op.set_params(baseline);

  const ParamSchemaDescriptor* const schema = param_schema_descriptor<CompileCatalogParams>();
  const ParamTypeDescriptor& type = param_type_descriptor<CompileCatalogParams>();
  const auto baseline_digest = detail::parameter_value_digest(type, schema, &baseline, false);
  ASSERT_TRUE(baseline_digest.available());
  const ParamFieldDescriptor& gain = catalog_field("gain");

  // Key setup: override changes only gain (2→5); secret and internal_counter are
  // unchanged from baseline and must survive the atomic application.
  CompileCatalogParams resolved = baseline;
  resolved.gain = 5U;
  const ParamSchemaHash resolved_digest = required_value_digest(resolved);
  std::vector<detail::ParamOverrideOperatorState> operators;
  operators.push_back(detail::ParamOverrideSetAccess::resolved_operator(
      "operator",
      resolved,
      baseline_digest.digest,
      resolved_digest,
      {{.field = gain.id,
        .full_id = gain.full_id,
        .origin = detail::ParamOverrideContributionOrigin::kCaptured}}));
  CompileOptions options;
  const ParamSchemaHash context = parameter_admission_context(graph, options);
  options.parameter_overrides = detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(), std::move(operators), context);

  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const detail::ExecutionPlanState* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->parameter_generation_zero_operators.size(), 1U);
  ASSERT_EQ(state->parameter_generation_zero_fields.size(), 3U);
  EXPECT_EQ(state->parameter_generation_zero_operators.front().baseline_origin,
            detail::ParameterGenerationZeroOrigin::kTypedBaseline);
  const auto overridden = std::find_if(state->parameter_generation_zero_fields.begin(),
                                       state->parameter_generation_zero_fields.end(),
                                       [&](const auto& field) { return field.field == gain.id; });
  ASSERT_NE(overridden, state->parameter_generation_zero_fields.end());
  EXPECT_EQ(overridden->origin, detail::ParameterGenerationZeroOrigin::kOverrideCaptured);

  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const void* const final_value =
      state->executable_factories.operators.front().factory->initial_parameter_value();
  ASSERT_NE(final_value, nullptr);
  // Key assertion: final value reflects the override (gain=5) while secret is retained.
  EXPECT_EQ(static_cast<const CompileCatalogParams*>(final_value)->gain, 5U);
  EXPECT_EQ(static_cast<const CompileCatalogParams*>(final_value)->secret, baseline.secret);
}

// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, RejectsDuplicateOverrideTargetsWithoutPartialFactoryReplacement) {
  Graph graph{"parameter-override-atomicity"};
  auto op = graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  const CompileCatalogParams baseline{.gain = 2U, .secret = 7U, .internal_counter = 9U};
  op.set_params(baseline);

  const ParamSchemaDescriptor* const schema = param_schema_descriptor<CompileCatalogParams>();
  const ParamTypeDescriptor& type = param_type_descriptor<CompileCatalogParams>();
  const auto baseline_digest = detail::parameter_value_digest(type, schema, &baseline, false);
  ASSERT_TRUE(baseline_digest.available());
  const ParamFieldDescriptor& gain = catalog_field("gain");
  const std::vector<detail::ParamOverrideFieldOriginRecord> contribution{{
      .field = gain.id,
      .full_id = gain.full_id,
      .origin = detail::ParamOverrideContributionOrigin::kCaptured,
  }};

  // Key setup: two resolved records for the same "operator" path, gain=3 and gain=4.
  CompileCatalogParams first = baseline;
  CompileCatalogParams second = baseline;
  first.gain = 3U;
  second.gain = 4U;
  std::vector<detail::ParamOverrideOperatorState> operators;
  operators.push_back(detail::ParamOverrideSetAccess::resolved_operator(
      "operator", first, baseline_digest.digest, required_value_digest(first), contribution));
  operators.push_back(detail::ParamOverrideSetAccess::resolved_operator(
      "operator", second, baseline_digest.digest, required_value_digest(second), contribution));
  CompileOptions options;
  const ParamSchemaHash context = parameter_admission_context(graph, options);
  options.parameter_overrides = detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(), std::move(operators), context);

  const ExecutionPlan plan = compile(graph, options);
  // Key assertion: duplicate target rejected; factory retains original baseline value.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_OVERRIDE_DUPLICATE_TARGET"));
  const detail::ExecutionPlanState* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const void* const final_value =
      state->executable_factories.operators.front().factory->initial_parameter_value();
  ASSERT_NE(final_value, nullptr);
  EXPECT_EQ(static_cast<const CompileCatalogParams*>(final_value)->gain, baseline.gain);
}

// REQ: COMPILE-019
// REQ: PARAM-007
// REQ: PARAM-009
TEST(ParameterCompile, RejectsMalformedUnknownAliasAndSchemaOverrideFactsDeterministically) {
  Graph graph{"parameter-override-invalid-facts"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  const auto compile_record = [&](detail::ParamOverrideOperatorState record) {
    CompileOptions options;
    std::vector<detail::ParamOverrideOperatorState> operators;
    operators.push_back(std::move(record));
    options.parameter_overrides = detail::ParamOverrideSetAccess::make_complete(
        graph.view().identity(), std::move(operators), parameter_admission_context(graph, options));
    return compile(graph, std::move(options));
  };

  // Key setup: five malformed override variants, each exercising one rejection path.
  auto malformed_path = catalog_gain_override_record(3U);
  malformed_path.operator_path = "operator//invalid";
  const ExecutionPlan malformed = compile_record(std::move(malformed_path));
  EXPECT_FALSE(malformed.ok());
  EXPECT_TRUE(test::has_diagnostic(malformed, "PARAMETER_OVERRIDE_TARGET_UNKNOWN"));

  auto unknown_field = catalog_gain_override_record(3U);
  unknown_field.contributions.front().field ^= 0x1U;
  const ExecutionPlan unknown = compile_record(std::move(unknown_field));
  EXPECT_FALSE(unknown.ok());
  EXPECT_TRUE(test::has_diagnostic(unknown, "PARAMETER_OVERRIDE_FIELD_INVALID"));

  auto invalid_alias = catalog_gain_override_record(3U);
  invalid_alias.contributions.front().origin = detail::ParamOverrideContributionOrigin::kAlias;
  const ExecutionPlan alias = compile_record(std::move(invalid_alias));
  EXPECT_FALSE(alias.ok());
  EXPECT_TRUE(test::has_diagnostic(alias, "PARAMETER_OVERRIDE_FIELD_INVALID"));

  auto wrong_schema_version = catalog_gain_override_record(3U);
  ++wrong_schema_version.schema_version;
  const ExecutionPlan schema = compile_record(std::move(wrong_schema_version));
  EXPECT_FALSE(schema.ok());
  EXPECT_TRUE(test::has_diagnostic(schema, "PARAMETER_OVERRIDE_SCHEMA_MISMATCH"));

  auto wrong_resolved_digest = catalog_gain_override_record(3U);
  wrong_resolved_digest.resolved_value_digest = {};
  const ExecutionPlan value = compile_record(std::move(wrong_resolved_digest));
  EXPECT_FALSE(value.ok());
  EXPECT_TRUE(test::has_diagnostic(value, "PARAMETER_OVERRIDE_VALUE_INVALID"));
}

// REQ: COMPILE-019
// REQ: PARAM-007
TEST(ParameterCompile, RejectsFieldPatchAgainstOpaqueTypedAggregate) {
  Graph graph{"parameter-override-opaque"};
  (void)graph.op<CompileParameterOperator<CompileOpaqueParams>>("operator");
  const CompileOpaqueParams resolved{.local_only = 2U};
  std::vector<detail::ParamOverrideOperatorState> operators;
  operators.push_back(detail::ParamOverrideSetAccess::resolved_operator(
      "operator",
      resolved,
      {},
      {},
      {{.field = 1U,
        .full_id = {},
        .origin = detail::ParamOverrideContributionOrigin::kCaptured}}));
  CompileOptions options;
  options.parameter_overrides = detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(), std::move(operators), parameter_admission_context(graph, options));

  // Key assertion: opaque type has no schema → field ID lookup fails with
  // PARAMETER_OVERRIDE_SCHEMA_MISMATCH and the initial value (local_only=1) is retained.
  const ExecutionPlan plan = compile(graph, std::move(options));
  EXPECT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_OVERRIDE_SCHEMA_MISMATCH"));
  EXPECT_TRUE(plan.parameter_fields().empty());
  EXPECT_TRUE(plan.parameter_catalog().fields().empty());
  const detail::ExecutionPlanState* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 1U);
  const void* const final_value =
      state->executable_factories.operators.front().factory->initial_parameter_value();
  ASSERT_NE(final_value, nullptr);
  EXPECT_EQ(static_cast<const CompileOpaqueParams*>(final_value)->local_only, 1U);
}

// REQ: COMPILE-002
// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, AdmissionContextIsIndependentOfDeploymentAuthoringOrder) {
  Graph graph{"parameter-admission-context-order"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  CompileOptions first;
  first.deployment.bind_thread_placement("alpha", ThreadPlacement{});
  first.deployment.bind_thread_placement("beta", ThreadPlacement{});
  CompileOptions second;
  second.deployment.bind_thread_placement("beta", ThreadPlacement{});
  second.deployment.bind_thread_placement("alpha", ThreadPlacement{});

  EXPECT_EQ(parameter_admission_context(graph, first), parameter_admission_context(graph, second));
}

// REQ: COMPILE-002
// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, AdmissionContextExcludesDiagnosticProducerBuildIdentity) {
  Graph graph{"parameter-admission-context-build"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  const CompileOptions options;
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  auto profile = detail::resolve_profile_admission(*snapshot, options.profile);
  profile.manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
  profile.manifest.target_triple = std::string(detail::compile_target_triple());
  std::vector partitions{ExecutionPartitionRecord{
      .id = ExecutionPartitionId{1U},
      .members = {snapshot->vertices.front().id},
      .execution_lane = 0U,
  }};
  const auto deployment = detail::lower_deployment_plan(
      *snapshot, options.deployment, profile.effective_profiles(), partitions);
  profile.manifest.producer_build_id = "diagnostic-build-a";
  const ParamSchemaHash first = detail::parameter_admission_context_digest(
      profile.manifest, deployment, detail::empty_parameter_migration_catalog_digest());
  profile.manifest.producer_build_id = "diagnostic-build-b";
  const ParamSchemaHash second = detail::parameter_admission_context_digest(
      profile.manifest, deployment, detail::empty_parameter_migration_catalog_digest());

  EXPECT_EQ(first, second);
}

// REQ: COMPILE-002
// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, OverrideSemanticDigestIsIndependentOfOperatorAuthoringOrder) {
  Graph graph{"parameter-override-semantic-order"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("a");
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("b");
  const CompileCatalogParams baseline{};
  const ParamFieldDescriptor& gain = catalog_field("gain");
  const auto make_record = [&](std::string path, std::uint32_t value) {
    CompileCatalogParams resolved = baseline;
    resolved.gain = value;
    return detail::ParamOverrideSetAccess::resolved_operator(
        std::move(path),
        resolved,
        required_value_digest(baseline),
        required_value_digest(resolved),
        {{.field = gain.id,
          .full_id = gain.full_id,
          .origin = detail::ParamOverrideContributionOrigin::kCaptured}});
  };
  const ParamSchemaHash context = parameter_admission_context(graph, CompileOptions{});
  // Key setup: identical overrides assembled in forward (a,b) and reverse (b,a) order.
  std::vector<detail::ParamOverrideOperatorState> forward;
  forward.push_back(make_record("a", 2U));
  forward.push_back(make_record("b", 3U));
  std::vector<detail::ParamOverrideOperatorState> reverse;
  reverse.push_back(make_record("b", 3U));
  reverse.push_back(make_record("a", 2U));

  const ParamOverrideSet first = detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(), std::move(forward), context);
  const ParamOverrideSet second = detail::ParamOverrideSetAccess::make_complete(
      graph.view().identity(), std::move(reverse), context);
  // Key assertion: digest is order-independent and matches the recomputed digest.
  EXPECT_EQ(detail::ParamOverrideSetAccess::semantic_digest(first),
            detail::ParamOverrideSetAccess::semantic_digest(second));
  EXPECT_EQ(detail::ParamOverrideSetAccess::semantic_digest(first),
            detail::ParamOverrideSetAccess::recompute_semantic_digest(first));
}

// REQ: COMPILE-002
// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, RejectsOverrideCapturedForDifferentProfileContext) {
  Graph graph{"parameter-override-profile-context"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  CompileOptions captured;
  captured.profile = Profile::kStandard;
  CompileOptions changed;
  changed.profile = Profile::kRobotics;
  changed.parameter_overrides = catalog_gain_override(graph, captured, 3U);

  const ExecutionPlan plan = compile(graph, changed);
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH"));
}

// REQ: COMPILE-002
// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, RejectsOverrideCapturedForDifferentDeploymentContext) {
  Graph graph{"parameter-override-deployment-context"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  const CompileOptions captured;
  CompileOptions changed;
  changed.deployment.bind_thread_placement("unused", ThreadPlacement{});
  changed.parameter_overrides = catalog_gain_override(graph, captured, 3U);

  const ExecutionPlan plan = compile(graph, changed);
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH"));
}

// REQ: COMPILE-002
// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, RejectsOverrideCapturedForDifferentMigrationCatalogContext) {
  Graph graph{"parameter-override-migration-context"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  CompileOptions options;
  detail::StableHashBuilder migration;
  migration.add("holoscan.test.nonempty_parameter_migration_catalog.v1");
  migration.add_canonical_size(1U);
  options.parameter_overrides = catalog_gain_override(graph, options, 3U, migration.finish());

  const ExecutionPlan plan = compile(graph, options);
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_OVERRIDE_ADMISSION_CONTEXT_MISMATCH"));
}

// REQ: COMPILE-002
// REQ: COMPILE-019
// REQ: PARAM-009
TEST(ParameterCompile, ResolvedOverrideValuesContributeToPlanGenerationIdentity) {
  Graph graph{"parameter-override-plan-generation"};
  (void)graph.op<CompileParameterOperator<CompileCatalogParams>>("operator");
  const CompileOptions context;

  CompileOptions first_options;
  first_options.parameter_overrides = catalog_gain_override(graph, context, 3U);
  CompileOptions second_options;
  second_options.parameter_overrides = catalog_gain_override(graph, context, 4U);

  const ExecutionPlan first = compile(graph, std::move(first_options));
  const ExecutionPlan second = compile(graph, std::move(second_options));
  ASSERT_TRUE(first.ok()) << first.json();
  ASSERT_TRUE(second.ok()) << second.json();
  const detail::ExecutionPlanState* const first_state = detail::ExecutionPlanAccess::state(first);
  const detail::ExecutionPlanState* const second_state = detail::ExecutionPlanAccess::state(second);
  ASSERT_NE(first_state, nullptr);
  ASSERT_NE(second_state, nullptr);
  EXPECT_NE(first_state->plan_generation, second_state->plan_generation);
}

// REQ: COMPILE-015
// REQ: COMPILE-060
// REQ: COMPILE-019
// REQ: PARAM-007
//
//   source --[queue_depth=2]--> target (CompileReplayOperator, gain overridden to 4)
//
TEST(ParameterCompile, ReplaysResolvedSetupAndContractWithoutMutatingAuthoringGraph) {
  Graph graph{"parameter-replay-positive"};
  test::add_compile_test_default_clock(graph);
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<CompileReplayOperator>("target");
  graph.add_flow(source->output, target->input);
  const GraphAuthoringIdentity identity = graph.view().identity();

  CompileReplayParams resolved;
  resolved.gain = 4U;
  CompileOptions options;
  options.parameter_overrides = replay_override(graph, options, resolved, "gain");
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key assertion: authoring graph identity is unchanged after compilation with override.
  EXPECT_EQ(graph.view().identity(), identity);
}

// REQ: COMPILE-019
// REQ: PARAM-007
//
//   source --[queue_depth=2]--> target (CompileReplayOperator, setup_shape overridden to 2)
//
TEST(ParameterCompile, RejectsParameterDependentSetupShapeBeforeFactoryPublication) {
  Graph graph{"parameter-replay-setup-negative"};
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<CompileReplayOperator>("target");
  graph.add_flow(source->output, target->input);

  CompileReplayParams resolved;
  resolved.setup_shape = 2U;
  CompileOptions options;
  options.parameter_overrides = replay_override(graph, options, resolved, "setup_shape");
  const ExecutionPlan plan = compile(graph, options);
  // Key assertion: port rename detected before factory publication → DECLARATION_MISMATCH;
  // factory retains the original setup_shape=1 value.
  ASSERT_FALSE(plan.ok());
  EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_OVERRIDE_DECLARATION_MISMATCH"));
  const detail::ExecutionPlanState* const state = detail::ExecutionPlanAccess::state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->executable_factories.operators.size(), 2U);
  const auto factory = std::find_if(state->executable_factories.operators.begin(),
                                    state->executable_factories.operators.end(),
                                    [](const auto& item) { return item.operator_id == 2U; });
  ASSERT_NE(factory, state->executable_factories.operators.end());
  EXPECT_EQ(static_cast<const CompileReplayParams*>(factory->factory->initial_parameter_value())
                ->setup_shape,
            1U);
}

// REQ: COMPILE-015
// REQ: COMPILE-019
// REQ: TEMPORAL-CONTRACT-001
//
//   source --[queue_depth=2]--> target (CompileReplayOperator, contract_valid=false)
//
TEST(ParameterCompile, RejectsInvalidContractFromResolvedGenerationZero) {
  Graph graph{"parameter-replay-contract-negative"};
  const auto source = graph.op<test::CompileTestOperator>("source");
  const auto target = graph.op<CompileReplayOperator>("target");
  graph.add_flow(source->output, target->input);

  CompileReplayParams resolved;
  resolved.contract_valid = false;
  CompileOptions options;
  options.parameter_overrides = replay_override(graph, options, resolved, "contract_valid");
  const ExecutionPlan plan = compile(graph, options);
  ASSERT_FALSE(plan.ok());
  // The override selects an undeclared Input member. Its PortRef has no
  // declaration name, so normalization rejects the contract before canonical
  // port-identity resolution is reached.
  EXPECT_TRUE(test::has_diagnostic(plan, "TEMPORAL_CONTRACT_INVALID")) << plan.json();
}

// REQ: COMPILE-003
// REQ: COMPILE-019
// REQ: PARAM-003
TEST(ParameterCompile, RequiresExplicitFieldPolicyEvidenceWhenRequestedOrRegulated) {
  // Key setup: strict mode (require_explicit_field_policies=true) and regulated profile
  // each independently require fully declared field policies.
  CompileOptions strict;
  strict.parameters.require_explicit_field_policies = true;
  const ExecutionPlan explicit_policy =
      compile_parameter_case<CompileExplicitPolicyParams>("parameter-explicit-policy", strict);
  ASSERT_TRUE(explicit_policy.ok()) << explicit_policy.json();
  EXPECT_FALSE(test::has_diagnostic(explicit_policy, "PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN"));

  const ExecutionPlan inferred_policy =
      compile_parameter_case<CompileInferredPolicyParams>("parameter-inferred-policy", strict);
  // Key assertion: inferred-policy schema fails strict mode; explicit-policy passes it.
  ASSERT_FALSE(inferred_policy.ok());
  EXPECT_TRUE(test::has_diagnostic(inferred_policy, "PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN"));

  CompileOptions regulated;
  regulated.profile = Profile::kRegulated;
  const ExecutionPlan regulated_explicit = compile_parameter_case<CompileExplicitPolicyParams>(
      "parameter-regulated-explicit-policy", regulated);
  EXPECT_FALSE(regulated_explicit.ok());
  EXPECT_TRUE(test::has_diagnostic(regulated_explicit, "PROFILE_EVIDENCE_UNSATISFIED"));
  EXPECT_FALSE(
      test::has_diagnostic(regulated_explicit, "PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN"));

  const ExecutionPlan regulated_inferred = compile_parameter_case<CompileInferredPolicyParams>(
      "parameter-regulated-inferred-policy", regulated);
  EXPECT_FALSE(regulated_inferred.ok());
  EXPECT_TRUE(test::has_diagnostic(regulated_inferred, "PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN"));
}

// REQ: COMPILE-003
// REQ: COMPILE-019
// REQ: COMPILE-030
// REQ: PARAM-002
TEST(ParameterCompile, AppliesOpaquePartialCompleteAndProofAdmissionMatrix) {
  // Key setup: four schema dispositions (opaque, partial, author-asserted, proven) tested
  // across Standard, Research, Robotics, RealTime, and Regulated profiles in one test.
  CompileOptions standard;
  standard.profile = Profile::kStandard;
  const ExecutionPlan opaque_standard =
      compile_parameter_case<CompileOpaqueParams>("parameter-opaque-standard", standard);
  ASSERT_TRUE(opaque_standard.ok()) << opaque_standard.json();
  ASSERT_EQ(opaque_standard.operator_parameters().size(), 1U);
  EXPECT_EQ(opaque_standard.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kOpaqueLocal);
  EXPECT_EQ(opaque_standard.operator_parameters().front().materialization,
            ParameterMaterializationStrategy::kColdOwnedTyped);
  EXPECT_TRUE(opaque_standard.parameter_fields().empty());
  EXPECT_TRUE(opaque_standard.parameter_catalog().fields().empty());

  const ExecutionPlan partial_standard =
      compile_parameter_case<CompilePartialParams>("parameter-partial-standard", standard);
  ASSERT_TRUE(partial_standard.ok()) << partial_standard.json();
  EXPECT_EQ(partial_standard.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kDeclaredPartial);
  EXPECT_EQ(partial_standard.operator_parameters().front().materialization,
            ParameterMaterializationStrategy::kColdOwnedTyped);

  CompileOptions research;
  research.profile = Profile::kResearch;
  const ExecutionPlan partial_research =
      compile_parameter_case<CompilePartialParams>("parameter-partial-research", research);
  ASSERT_TRUE(partial_research.ok()) << partial_research.json();
  EXPECT_EQ(partial_research.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kDeclaredPartial);

  const ExecutionPlan asserted_standard = compile_parameter_case<CompileAuthorAssertedParams>(
      "parameter-author-asserted-standard", standard);
  ASSERT_TRUE(asserted_standard.ok()) << asserted_standard.json();
  EXPECT_EQ(asserted_standard.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kCompleteAuthorAsserted);
  EXPECT_EQ(asserted_standard.operator_parameters().front().materialization,
            ParameterMaterializationStrategy::kFixedObject);

  CompileOptions robotics;
  robotics.profile = Profile::kRobotics;
  const ExecutionPlan opaque_robotics =
      compile_parameter_case<CompileOpaqueParams>("parameter-opaque-robotics", robotics);
  // Key assertion: Robotics/RealTime/Regulated profiles reject opaque and partial schemas.
  EXPECT_FALSE(opaque_robotics.ok()) << opaque_robotics.json();
  EXPECT_TRUE(test::has_diagnostic(opaque_robotics, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED"));
  EXPECT_EQ(opaque_robotics.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kOpaqueLocal);
  EXPECT_EQ(opaque_robotics.operator_parameters().front().materialization,
            ParameterMaterializationStrategy::kRejected);

  const ExecutionPlan partial_robotics =
      compile_parameter_case<CompilePartialParams>("parameter-partial-robotics", robotics);
  EXPECT_FALSE(partial_robotics.ok()) << partial_robotics.json();
  EXPECT_TRUE(test::has_diagnostic(partial_robotics, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED"));
  EXPECT_EQ(partial_robotics.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kDeclaredPartial);
  EXPECT_EQ(partial_robotics.operator_parameters().front().materialization,
            ParameterMaterializationStrategy::kRejected);

  CompileOptions require_proven = standard;
  require_proven.parameters.require_proven_schema_completeness = true;
  const ExecutionPlan asserted_strict = compile_parameter_case<CompileAuthorAssertedParams>(
      "parameter-author-asserted-strict", require_proven);
  EXPECT_FALSE(asserted_strict.ok());
  EXPECT_TRUE(test::has_diagnostic(asserted_strict, "PARAMETER_SCHEMA_INCOMPLETE"));

  CompileOptions real_time;
  real_time.profile = Profile::kRealTime;
  const ExecutionPlan partial_real_time =
      compile_parameter_case<CompilePartialParams>("parameter-partial-real-time", real_time);
  EXPECT_FALSE(partial_real_time.ok());
  EXPECT_TRUE(test::has_diagnostic(partial_real_time, "PARAMETER_SCHEMA_INCOMPLETE"));
  EXPECT_EQ(partial_real_time.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kRejectedByProfile);

  const ExecutionPlan complete_real_time = compile_parameter_case<CompileExplicitPolicyParams>(
      "parameter-complete-real-time", real_time);
  EXPECT_FALSE(complete_real_time.ok());
  EXPECT_TRUE(test::has_diagnostic(complete_real_time, "PROFILE_CAPABILITY_UNSATISFIED"))
      << complete_real_time.json();
  EXPECT_FALSE(test::has_diagnostic(complete_real_time, "PARAMETER_SCHEMA_INCOMPLETE"));
  EXPECT_EQ(complete_real_time.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kCompleteProven);
  EXPECT_EQ(complete_real_time.operator_parameters().front().materialization,
            ParameterMaterializationStrategy::kFixedObject);

  CompileOptions regulated;
  regulated.profile = Profile::kRegulated;
  const ExecutionPlan complete_regulated = compile_parameter_case<CompileExplicitPolicyParams>(
      "parameter-complete-regulated", regulated);
  EXPECT_FALSE(complete_regulated.ok());
  EXPECT_TRUE(test::has_diagnostic(complete_regulated, "PROFILE_EVIDENCE_UNSATISFIED"))
      << complete_regulated.json();
  EXPECT_FALSE(test::has_diagnostic(complete_regulated, "PARAMETER_SCHEMA_INCOMPLETE"));
  EXPECT_FALSE(
      test::has_diagnostic(complete_regulated, "PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN"));
  EXPECT_EQ(complete_regulated.operator_parameters().front().schema_disposition,
            ParameterSchemaAdmissionDisposition::kCompleteProven);
}

// REQ: COMPILE-003
// REQ: COMPILE-019
// REQ: PARAM-002
// REQ: COMPILE-030
TEST(ParameterCompile, MakesDynamicStringVectorAndPathMaterializationExplicit) {
  // Key setup: three dynamic-storage types (string, vector, path) tested across Standard,
  // Robotics, RealTime, and Regulated profiles plus inline-only and preallocated modes.
  for (const Profile profile : {Profile::kStandard, Profile::kResearch}) {
    CompileOptions options;
    options.profile = profile;
    const ExecutionPlan string_plan =
        compile_parameter_case<CompileDynamicParams>("parameter-dynamic-string", options);
    const ExecutionPlan vector_plan =
        compile_parameter_case<CompileVectorParams>("parameter-dynamic-vector", options);
    const ExecutionPlan path_plan =
        compile_parameter_case<CompilePathParams>("parameter-dynamic-path", options);
    ASSERT_TRUE(string_plan.ok()) << string_plan.json();
    ASSERT_TRUE(vector_plan.ok()) << vector_plan.json();
    ASSERT_TRUE(path_plan.ok()) << path_plan.json();
    EXPECT_EQ(string_plan.operator_parameters().front().materialization,
              ParameterMaterializationStrategy::kColdOwnedTyped);
    EXPECT_EQ(vector_plan.operator_parameters().front().materialization,
              ParameterMaterializationStrategy::kColdOwnedTyped);
    EXPECT_EQ(path_plan.operator_parameters().front().materialization,
              ParameterMaterializationStrategy::kColdOwnedTyped);
    // Key assertion: all three types are kColdOwnedTyped; string owns dynamic storage
    // and does not support bounded materialization.
    EXPECT_TRUE(string_plan.operator_parameters().front().storage.owns_dynamic_storage);
    EXPECT_FALSE(
        string_plan.operator_parameters().front().storage.supports_bounded_materialization);
  }

  CompileOptions robotics;
  robotics.profile = Profile::kRobotics;
  for (const ExecutionPlan& plan : {
           compile_parameter_case<CompileDynamicParams>("parameter-robotics-string", robotics),
           compile_parameter_case<CompileVectorParams>("parameter-robotics-vector", robotics),
           compile_parameter_case<CompilePathParams>("parameter-robotics-path", robotics),
       }) {
    EXPECT_FALSE(plan.ok());
    EXPECT_TRUE(test::has_diagnostic(plan, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED"));
    ASSERT_FALSE(plan.operator_parameters().empty());
    EXPECT_EQ(plan.operator_parameters().front().materialization,
              ParameterMaterializationStrategy::kRejected);
  }

  for (const ParamStorageMode storage :
       {ParamStorageMode::kDynamicColdPath, ParamStorageMode::kFreezeBeforeArm}) {
    CompileOptions options;
    options.parameters.storage = storage;
    const ExecutionPlan path_plan =
        compile_parameter_case<CompilePathParams>("parameter-dynamic-path-storage", options);
    ASSERT_TRUE(path_plan.ok()) << path_plan.json();
    EXPECT_EQ(path_plan.operator_parameters().front().materialization,
              ParameterMaterializationStrategy::kColdOwnedTyped);
  }

  CompileOptions inline_only;
  inline_only.parameters.storage = ParamStorageMode::kInlineOnly;
  const ExecutionPlan bounded_string =
      compile_parameter_case<CompileDynamicParams>("parameter-bounded-string-inline", inline_only);
  EXPECT_FALSE(bounded_string.ok());
  EXPECT_TRUE(test::has_diagnostic(bounded_string, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED"));

  CompileOptions preallocated;
  preallocated.parameters.storage = ParamStorageMode::kPreallocatedBounded;
  const ExecutionPlan bounded_vector = compile_parameter_case<CompileVectorParams>(
      "parameter-bounded-vector-preallocated", preallocated);
  EXPECT_FALSE(bounded_vector.ok());
  EXPECT_TRUE(test::has_diagnostic(bounded_vector, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED"));

  for (const Profile profile : {Profile::kRealTime, Profile::kRegulated}) {
    CompileOptions options;
    options.profile = profile;
    options.parameters.storage = ParamStorageMode::kFreezeBeforeArm;
    const ExecutionPlan string_plan =
        compile_parameter_case<CompileDynamicParams>("parameter-constrained-string", options);
    const ExecutionPlan vector_plan =
        compile_parameter_case<CompileVectorParams>("parameter-constrained-vector", options);
    const ExecutionPlan path_plan =
        compile_parameter_case<CompilePathParams>("parameter-constrained-path", options);
    for (const ExecutionPlan* plan : {&string_plan, &vector_plan, &path_plan}) {
      EXPECT_FALSE(plan->ok());
      EXPECT_TRUE(test::has_diagnostic(*plan, "PARAMETER_UPDATE_STORAGE_UNSUPPORTED"));
      EXPECT_EQ(plan->operator_parameters().front().materialization,
                ParameterMaterializationStrategy::kRejected);
    }
  }
}

}  // namespace
}  // namespace holoscan
// NOLINTEND(bugprone-unchecked-optional-access)
