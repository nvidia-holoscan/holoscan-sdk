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
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/parameter_generation_zero.hpp"
#include "compiler/parameter_identity.hpp"
#include "compiler/runtime_package_capabilities.hpp"
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
        holoscan::param_schema("holoscan.test.CompileCatalogParams", 1)),
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

HOLOSCAN_PARAM_SCHEMA(CompilePartialParams,
                      holoscan::param_schema("holoscan.test.CompilePartialParams", 1).partial(),
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
                      holoscan::param_schema("holoscan.test.CompileAuthorAssertedParams", 1),
                      holoscan::field<&CompileAuthorAssertedParams::value>("value"));

// UTF-8 string field (max 32 bytes): dynamic heap storage forces kColdOwnedTyped
// materialization and blocks bounded and inline-only storage modes.
// Representative test: MakesDynamicStringVectorAndPathMaterializationExplicit.
struct CompileDynamicParams {
  std::string label{"camera"};
};

HOLOSCAN_PARAM_SCHEMA(CompileDynamicParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompileDynamicParams", 1)),
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
                          holoscan::param_schema("holoscan.test.CompileVectorParams", 1)),
                      holoscan::field<&CompileVectorParams::values>("values").max_elements(8U));

// Optional scalar field: its underlying integer shape remains discoverable, but absence makes
// the baseline fixed-scalar control route inapplicable (REQ: PARAM-054).
struct CompileOptionalParams {
  std::optional<std::uint32_t> value{4U};
};

HOLOSCAN_PARAM_SCHEMA(CompileOptionalParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompileOptionalParams", 1)),
                      holoscan::field<&CompileOptionalParams::value>("value").range(1U, 8U));

// Explicitly named enum values remain user-facing metadata in the plan and catalog rather than
// compiler-derived C++ identifiers (REQ: PARAM-055).
enum class CompileMode : std::uint8_t {
  kDisabled = 0,
  kAutomatic = 2,
  kManual = 7,
};

struct CompileNamedEnumParams {
  CompileMode mode{CompileMode::kAutomatic};
};

HOLOSCAN_PARAM_SCHEMA(CompileNamedEnumParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompileNamedEnumParams", 1)),
                      holoscan::field<&CompileNamedEnumParams::mode>("mode").enum_values(
                          {param_enum_value("disabled", CompileMode::kDisabled),
                           param_enum_value("automatic", CompileMode::kAutomatic),
                           param_enum_value("manual", CompileMode::kManual)}));

// std::filesystem::path field: always cold-owned regardless of storage mode.
// Representative test: MakesDynamicStringVectorAndPathMaterializationExplicit.
struct CompilePathParams {
  std::filesystem::path value{"model.plan"};
};

HOLOSCAN_PARAM_SCHEMA(CompilePathParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompilePathParams", 1)),
                      holoscan::field<&CompilePathParams::value>("value"));

// Proven schema where persistence, access, and sensitivity are left at their inferred
// defaults; fails strict-mode compilation with PARAMETER_EXPLICIT_FIELD_POLICY_UNPROVEN.
// Representative test: RequiresExplicitFieldPolicyEvidenceWhenRequestedOrRegulated.
struct CompileInferredPolicyParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileInferredPolicyParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompileInferredPolicyParams", 1)),
                      holoscan::field<&CompileInferredPolicyParams::value>("value"));

// Proven schema with every field declaring persistence, access, and sensitivity
// explicitly; passes strict-mode and regulated-profile compilation.
// Representative test: RequiresExplicitFieldPolicyEvidenceWhenRequestedOrRegulated.
struct CompileExplicitPolicyParams {
  std::uint32_t value{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileExplicitPolicyParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompileExplicitPolicyParams", 1)),
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
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompileExplicitRuntimeParams", 1)),
                      holoscan::field<&CompileExplicitRuntimeParams::value>("value")
                          .runtime_mutable()
                          .persistence(ParamPersistence::kPersisted)
                          .external_access(ParamExternalAccess::kTuning)
                          .sensitivity(ParamSensitivity::kNormal));

// Single gain field admitted for lifecycle-stage mutation (kConfigure).
// Representative test: LifecycleOptInAdmitsPrivateCandidateWithoutComputeIngress.
struct CompileLifecycleParams {
  std::uint32_t gain{1U};
};

HOLOSCAN_PARAM_SCHEMA(CompileLifecycleParams,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.CompileLifecycleParams", 1)),
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
// Graph topology:
//   operator (isolated Parameter-bearing Operator)
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

// Compiles a single CompileParameterOperator<ParamsT> named "operator".
template <typename ParamsT>
[[nodiscard]] ExecutionPlan compile_parameter_case(std::string_view graph_name,
                                                   CompileOptions options = {}) {
  Graph graph{graph_name};
  (void)graph.op<CompileParameterOperator<ParamsT>>("operator");
  return compile(graph, std::move(options));
}

TEST(ParameterCompile, BuildsDeterministicPolicyFilteredCatalogAndFiniteCapacity) {
  // Graph topology:
  //   operator (isolated Parameter-bearing Operator)
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

// REQ: PARAM-054
TEST(ParameterCompile, PreservesOptionalShapeWithoutCollapsingAbsenceIntoScalarRouting) {
  const ExecutionPlan plan = compile_parameter_case<CompileOptionalParams>("parameter-optional");
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.parameter_fields().size(), 1U);

  const ParameterFieldPlanRecord& planned = plan.parameter_fields().front();
  EXPECT_EQ(planned.logical_type, ParamLogicalType::kUnsignedInteger);
  EXPECT_TRUE(planned.type_shape.optional);
  EXPECT_EQ(planned.type_shape.bit_width, 32U);
  EXPECT_TRUE(planned.type_shape.has_stable_shape);
  EXPECT_EQ(planned.value_route, ParamValueRouteDisposition::kLocalTypedOnly);
  EXPECT_EQ(planned.materialization, ParameterMaterializationStrategy::kColdOwnedTyped);

  const ParameterCatalogView catalog = plan.parameter_catalog();
  ASSERT_EQ(catalog.fields().size(), 1U);
  EXPECT_TRUE(catalog.fields().front().type_shape.optional);
  EXPECT_EQ(catalog.fields().front().value_route, ParamValueRouteDisposition::kLocalTypedOnly);
  EXPECT_NE(plan.json().find("\"optional\":true"), std::string::npos);
}

// REQ: PARAM-055
TEST(ParameterCompile, PreservesNamedEnumChoicesInPlanCatalogAndArtifact) {
  const ExecutionPlan plan = compile_parameter_case<CompileNamedEnumParams>("parameter-named-enum");
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.parameter_fields().size(), 1U);

  const ParameterFieldPlanRecord& planned = plan.parameter_fields().front();
  ASSERT_EQ(planned.enum_choices.size(), 3U);
  EXPECT_EQ(planned.enum_choices[0].name, "disabled");
  EXPECT_EQ(planned.enum_choices[1].name, "automatic");
  EXPECT_EQ(planned.enum_choices[2].name, "manual");
  CompileMode decoded{};
  EXPECT_TRUE(planned.enum_choices[1].value.try_as(decoded));
  EXPECT_EQ(decoded, CompileMode::kAutomatic);
  EXPECT_EQ(planned.value_route, ParamValueRouteDisposition::kPortableScalar);

  const ParameterCatalogView catalog = plan.parameter_catalog();
  ASSERT_EQ(catalog.fields().size(), 1U);
  ASSERT_EQ(catalog.fields().front().enum_choices.size(), 3U);
  EXPECT_EQ(catalog.fields().front().enum_choices[1].name, "automatic");

  const std::string_view json = plan.json();
  EXPECT_NE(json.find("\"enum_choices\":[{\"name\":\"disabled\""), std::string::npos);
  EXPECT_NE(json.find("\"name\":\"automatic\""), std::string::npos);
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
  // Graph topology:
  //   operator (isolated lifecycle-Parameter Operator)
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
  // Graph topology:
  //   operator (isolated lifecycle-Parameter Operator)
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
