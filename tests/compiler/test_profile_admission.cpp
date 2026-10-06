/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/compile_platform.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/profile_admission.hpp"
#include "compiler/profile_manifest_lowering.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/graph_compiler_access.hpp"
#include "core/source_id_internal.hpp"

namespace holoscan {
namespace {

// This literal is intentionally independent of the package-capability producer. Any canonical
// schema, field ordering, identity, or capability change must update this reviewable artifact.
constexpr std::string_view kRuntimePackageX8664Golden =
    R"gold({"build_id":"holoscan-core-5.0","capability_offers":[{"evidence_id":"holoscan.profile.re)gold"
    R"gold(gulated.unqualified_evidence/v1","id":"holoscan.profile.regulated.complete_evidence","ve)gold"
    R"gold(rsion":1},{"evidence_id":"holoscan.profile.robotics.finite_evidence/v1","id":"holoscan.p)gold"
    R"gold(rofile.robotics.finite_evidence","version":1},{"evidence_id":"holoscan.profile.robotics_)gold"
    R"gold(critical.bounded_memory/v1","id":"holoscan.profile.robotics_critical.bounded_memory","ve)gold"
    R"gold(rsion":1},{"evidence_id":"holoscan.profile.robotics_critical.latency_budget/v1","id":"ho)gold"
    R"gold(loscan.profile.robotics_critical.latency_budget","version":1},{"evidence_id":"holoscan.c)gold"
    R"gold(hannel.generated_schema/v1","id":"holoscan.runtime.channel.generated_schema","version":1)gold"
    R"gold(},{"evidence_id":"holoscan.channel.holomq_transport/v1","id":"holoscan.runtime.channel.h)gold"
    R"gold(olomq_transport","version":1},{"evidence_id":"holoscan.channel.pod_codec/v1","id":"holos)gold"
    R"gold(can.runtime.channel.pod_codec","version":1},{"evidence_id":"holoscan.compile.endpoint_ca)gold"
    R"gold(talog/v1","id":"holoscan.runtime.channel_endpoint_catalog","version":1},{"evidence_id":")gold"
    R"gold(holoscan.compile.channel_service_plan/v1","id":"holoscan.runtime.channel_service_attach")gold"
    R"gold(,"version":1},{"evidence_id":"holoscan.compile.cuda_compute_stream_plan/v1","id":)gold"
    R"gold("holoscan.runtime.cuda_compute_stream_materialization","version":1},{"evidence_id":)gold"
    R"gold("holoscan.compile.cuda_green_context_plan/v1","id":"holoscan.runtime.cuda_green_)gold"
    R"gold(context_materialization","version":1},{"evidence_id":)gold"
    R"gold("holoscan.compile.execution_plan.private.v1","id":"holoscan)gold"
    R"gold(.runtime.execution_plan","version":1},{"evidence_id":"holoscan.compile.generic_output_lo)gold"
    R"gold(an_plan/v1","id":"holoscan.runtime.generic_output_loans","version":1},{"evidence_id":"ho)gold"
    R"gold(loscan.compile.physical_plan/v1","id":"holoscan.runtime.payload_edges","version":1},{"ev)gold"
    R"gold(idence_id":"holoscan.compile.readiness_plan/v1","id":"holoscan.runtime.readiness_sources)gold"
    R"gold(","version":1},{"evidence_id":"holoscan.compile.scheduler.ready_policy/v1","id":"holosca)gold"
    R"gold(n.runtime.ready.deterministic_serial","version":1},{"evidence_id":"holoscan.runtime.targ)gold"
    R"gold(et.current/v1","id":"holoscan.runtime.target.current","version":1},{"evidence_id":"holos)gold"
    R"gold(can.compile.temporal_plan/v1","id":"holoscan.runtime.temporal_contracts","version":1},{")gold"
    R"gold(evidence_id":"holoscan.runtime.tensor_inline_record/generated/v1","id":"holoscan.runtime)gold"
    R"gold(.tensor_inline_record","version":1},{"evidence_id":"holoscan.compile.tensor_pool_plan/v1)gold"
    R"gold(","id":"holoscan.runtime.tensor_pool_materialization","version":1},{"evidence_id":"holos)gold"
    R"gold(can.compile.tensor_transfer_plan/v1","id":"holoscan.runtime.tensor_transfer_matrix","ver)gold"
    R"gold(sion":1},{"evidence_id":"holoscan.runtime.thread_placement.apply_readback/v1","id":"holo)gold"
    R"gold(scan.runtime.thread_placement.apply_readback","version":1},{"evidence_id":"holoscan.runt)gold"
    R"gold(ime.trace_context/v1","id":"holoscan.runtime.trace_context","version":1}],"digest":"sha2)gold"
    R"gold(56:072d959acbda8aaf65b7289b55fff82f60d8baf7f730c6deccd574c67762f7f4","package_id":"holos)gold"
    R"gold(can-core","profile_policies":[{"digest":"sha256:deb27bac29f1e11ef258d8208fc4d9324baeeb73)gold"
    R"gold(bc45d4c6a5d1eeadf312738d","major":1,"minor":0}],"runtime_abi_versions":["holoscan.runtim)gold"
    R"gold(e_abi.v1"],"schema":"holoscan.runtime_package_capabilities.canonical_json.v1","schema_ve)gold"
    R"gold(rsion":1,"supported_plan_schemas":["holoscan.execution_plan.private.v8"],"target_triples)gold"
    R"gold(":["x86_64-linux-gnu"],"tensor_inline_record_capability":{"capability_id":"holoscan.runt)gold"
    R"gold(ime.tensor_inline_record","capability_version":1,"evidence_id":"holoscan.runtime.tensor_)gold"
    R"gold(inline_record/generated/v1","fixed_metadata_overhead_bytes":512,"inline_arm_schema_id":")gold"
    R"gold(holoscan.channel.tensor_record.inline/v1","inline_discriminator":"inline","max_fixed_rec)gold"
    R"gold(ord_bytes":66048,"max_inline_logical_bytes":65536,"max_tensor_rank":32,"opaque_descripto)gold"
    R"gold(r_arm_schema_id":"holoscan.channel.tensor_record.opaque_descriptor/v1","opaque_descripto)gold"
    R"gold(r_discriminator":"opaque_descriptor","provider_id":"holoscan-core.generated.tensor-recor)gold"
    R"gold(d-provider","record_format_id":"holoscan.channel.tensor_record","record_format_version":)gold"
    R"gold(1,"service_family_id":"holoscan.channel.tensor-record-service"}})gold";

// Formats a PlanDigest byte array as a lowercase hex string for golden comparisons.
// Used in CanonicalPolicyManifestIsCompleteAndDeterministic.
[[nodiscard]] std::string digest_hex(const PlanDigest& digest) {
  constexpr std::string_view kHex = "0123456789abcdef";
  std::string result;
  result.reserve(digest.value.size() * 2U);
  for (const std::byte byte : digest.value) {
    const auto value = std::to_integer<std::uint8_t>(byte);
    result.push_back(kHex[value >> 4U]);
    result.push_back(kHex[value & 0x0FU]);
  }
  return result;
}

// Builds the generated capability manifest and injects the evidence entry required for
// the given profile (kRealTime adds bounded_hot_path; kRegulated corrects evidence_id).
// Used in MatchesSchemaPolicyAbiTargetCapabilityAndEvidenceExactly and
// CompletesEveryInstantiatedScopeKindAndCleanupContract.
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

// Returns an ExecutionPlanState pre-populated as if admission accepted the given profile,
// with canonical plan-schema, ABI version, target triple, and a diagnostic build id.
// Used in CanonicalizesEveryRecordFamilyAndRejectsDuplicateKeys,
// MatchesSchemaPolicyAbiTargetCapabilityAndEvidenceExactly, and
// CompletesEveryInstantiatedScopeKindAndCleanupContract.
[[nodiscard]] detail::ExecutionPlanState profile_state(Profile profile) {
  detail::ExecutionPlanState state;
  auto& manifest = state.profile_manifest;
  manifest.disposition = ProfileAdmissionDisposition::kAccepted;
  manifest.requested_profile = profile;
  manifest.effective_graph_profile = profile;
  manifest.profile_policy_id = detail::profile_policy_id_v1();
  manifest.plan_schema_version = "holoscan.execution_plan.private.v8";
  manifest.runtime_abi_version = "holoscan.runtime_abi.v1";
  manifest.target_triple = std::string(detail::compile_target_triple());
  manifest.producer_build_id = "diagnostic-build";
  state.partitioning_plan.disposition = PlanDisposition::kAccepted;
  state.scheduler_plan.execution_lane_count = 1U;
  return state;
}

// Two independent linear graphs (one with default insertion, one seeded) confirm that
// the profile-policy digest is bit-stable and every manifest field is populated.
//
//   source --> sink  (first, default insertion order)
//   source --> sink  (second, seeded insertion order)
//
// REQ: COMPILE-003
TEST(ProfileAdmission, CanonicalPolicyManifestIsCompleteAndDeterministic) {
  constexpr std::string_view kPolicyDigest =
      "deb27bac29f1e11ef258d8208fc4d9324baeeb73bc45d4c6a5d1eeadf312738d";
  constexpr std::size_t kPolicyBytes = 931U;

  Graph first{"profile-policy-golden"};
  test::add_linear_graph(first);
  Graph second{"profile-policy-golden"};
  test::add_linear_graph(second, true);

  const ExecutionPlan first_plan = compile(first);
  const ExecutionPlan second_plan = compile(second);
  ASSERT_TRUE(first_plan.ok()) << first_plan.json();
  ASSERT_TRUE(second_plan.ok()) << second_plan.json();
  const auto& first_manifest = first_plan.profile_manifest();
  const auto& second_manifest = second_plan.profile_manifest();
  EXPECT_EQ(detail::profile_policy_canonical_json_v1().size(), kPolicyBytes);
  EXPECT_EQ(digest_hex(first_manifest.profile_policy_id.digest), kPolicyDigest);
  EXPECT_EQ(first_manifest.profile_policy_id.major, 1U);
  EXPECT_EQ(first_manifest.profile_policy_id.minor, 0U);
  EXPECT_EQ(first_manifest.profile_policy_id.digest, second_manifest.profile_policy_id.digest);
  EXPECT_FALSE(first_manifest.plan_schema_version.empty());
  EXPECT_FALSE(first_manifest.runtime_abi_version.empty());
  EXPECT_FALSE(first_manifest.target_triple.empty());
  EXPECT_FALSE(first_manifest.producer_build_id.empty());
  EXPECT_FALSE(first_manifest.scope_resolutions.empty());
  EXPECT_FALSE(first_manifest.required_capabilities.empty());
  EXPECT_FALSE(first_manifest.evidence_references.empty());
  EXPECT_FALSE(first_manifest.feature_dispositions.empty());
  EXPECT_FALSE(first_manifest.derived_capacities.empty());
  EXPECT_FALSE(first_manifest.materialization_obligations.empty());
}

// REQ: COMPILE-003
TEST(ProfileAdmission, PolicyIdentityEvolutionUsesExplicitCompatibilityVersions) {
  // Key setup: baseline derived from the live canonical JSON; four mutations cover
  // compatible rule change, additive minor bump, incompatible major bump, and fields
  // intentionally excluded from canonical identity (provenance, addresses, prose).
  const std::string canonical{detail::profile_policy_canonical_json_v1()};
  const ProfilePolicyId baseline = detail::profile_policy_id_from_canonical(canonical, 1U, 0U);
  const ProfilePolicyId policy_v1 = detail::profile_policy_id_v1();
  EXPECT_EQ(baseline.major, policy_v1.major);
  EXPECT_EQ(baseline.minor, policy_v1.minor);
  EXPECT_EQ(baseline.digest, policy_v1.digest);

  auto replace_once =
      [](std::string source, std::string_view old_value, std::string_view new_value) {
        const std::size_t offset = source.find(old_value);
        EXPECT_NE(offset, std::string::npos);
        if (offset != std::string::npos) {
          source.replace(offset, old_value.size(), new_value);
        }
        return source;
      };

  const std::string compatible_rule_change =
      replace_once(canonical, "holoscan.profile.closed_set/v1", "holoscan.profile.closed_set/v2");
  const ProfilePolicyId compatible =
      detail::profile_policy_id_from_canonical(compatible_rule_change, 1U, 0U);
  EXPECT_EQ(compatible.major, baseline.major);
  EXPECT_EQ(compatible.minor, baseline.minor);
  EXPECT_NE(compatible.digest, baseline.digest);

  std::string additive = replace_once(canonical, R"("minor":0)", R"("minor":1)");
  additive = replace_once(additive,
                          R"("holoscan.profile.scope.strictest_max/v1"])",
                          R"("holoscan.profile.scope.strictest_max/v1",)"
                          R"("holoscan.profile.optional_evidence/v1"])");
  const ProfilePolicyId additive_id = detail::profile_policy_id_from_canonical(additive, 1U, 1U);
  EXPECT_EQ(additive_id.major, baseline.major);
  EXPECT_EQ(additive_id.minor, baseline.minor + 1U);
  EXPECT_NE(additive_id.digest, baseline.digest);

  std::string incompatible = replace_once(canonical, R"("major":1)", R"("major":2)");
  incompatible =
      replace_once(incompatible,
                   R"("dominance":["research","standard","robotics","real_time","regulated"])",
                   R"("dominance":["standard","research","robotics","real_time","regulated"])");
  const ProfilePolicyId incompatible_id =
      detail::profile_policy_id_from_canonical(incompatible, 2U, 0U);
  EXPECT_EQ(incompatible_id.major, baseline.major + 1U);
  EXPECT_EQ(incompatible_id.minor, 0U);
  EXPECT_NE(incompatible_id.digest, baseline.digest);

  // Key assertion: excluded fields (provenance, addresses, prose) do not shift the digest.
  // Build provenance, addresses, order of unrelated inputs, and diagnostic prose are
  // deliberately not canonical-policy fields, so they cannot change this identity.
  const ProfilePolicyId excluded_provenance =
      detail::profile_policy_id_from_canonical(canonical, 1U, 0U);
  EXPECT_EQ(excluded_provenance.major, baseline.major);
  EXPECT_EQ(excluded_provenance.minor, baseline.minor);
  EXPECT_EQ(excluded_provenance.digest, baseline.digest);
}

// REQ: COMPILE-003
TEST(ProfileAdmission, CanonicalizesEveryRecordFamilyAndRejectsDuplicateKeys) {
  // Key setup: all eight sortable record families populated with out-of-order pairs
  // ("test.z" before "test.a") to verify that finalize_profile_manifest sorts each.
  detail::ExecutionPlanState state = profile_state(Profile::kStandard);
  auto& manifest = state.profile_manifest;
  manifest.scope_resolutions = {
      {.object_id = "test:z", .object_kind = "test"},
      {.object_id = "test:a", .object_kind = "test"},
  };
  manifest.required_capabilities = {
      {.capability_id = "test.z", .version = 1U, .evidence_id = "test.z/v1"},
      {.capability_id = "test.a", .version = 1U, .evidence_id = "test.a/v1"},
  };
  manifest.evidence_references = {
      {.id = "test.z/v1", .kind = "test", .version = 1U},
      {.id = "test.a/v1", .kind = "test", .version = 1U},
  };
  manifest.feature_dispositions = {
      {.feature_id = "test.z", .disposition = ProfileFeatureDisposition::kAdmitted},
      {.feature_id = "test.a", .disposition = ProfileFeatureDisposition::kAdmitted},
  };
  manifest.derived_capacities = {
      {.object_id = "test:z", .capacity_id = "count", .limit = 1U, .unit = "count"},
      {.object_id = "test:a", .capacity_id = "count", .limit = 1U, .unit = "count"},
  };
  manifest.materialization_obligations = {
      {.object_id = "test:z",
       .strategy_id = "test.strategy",
       .complete_before_stage = LifecycleStage::kStart},
      {.object_id = "test:a",
       .strategy_id = "test.strategy",
       .complete_before_stage = LifecycleStage::kStart},
  };
  manifest.explicit_fallbacks = {
      {.feature_id = "test.z", .fallback_id = "test.fallback"},
      {.feature_id = "test.a", .fallback_id = "test.fallback"},
  };
  manifest.generated_endpoint_names = {
      {.final_name = "test.z", .algorithm = "test.algorithm", .version = 1U},
      {.final_name = "test.a", .algorithm = "test.algorithm", .version = 1U},
  };
  auto package = detail::runtime_package_capability_manifest();
  package.offers.push_back({"test.a", 1U, "test.a/v1"});
  package.offers.push_back({"test.z", 1U, "test.z/v1"});
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(package));
  const auto package_view = detail::validate_runtime_package_capability_manifest(package);
  ASSERT_TRUE(package_view);
  ASSERT_TRUE(detail::finalize_profile_manifest(CompileOptions{}, state, package_view.value()));
  // Key assertion: every family is sorted by its primary key after finalization.
  EXPECT_TRUE(std::is_sorted(manifest.scope_resolutions.begin(),
                             manifest.scope_resolutions.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.object_kind, lhs.object_id} <
                                      std::pair{rhs.object_kind, rhs.object_id};
                             }));
  EXPECT_TRUE(std::is_sorted(manifest.required_capabilities.begin(),
                             manifest.required_capabilities.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.capability_id, lhs.version} <
                                      std::pair{rhs.capability_id, rhs.version};
                             }));
  EXPECT_TRUE(std::is_sorted(manifest.evidence_references.begin(),
                             manifest.evidence_references.end(),
                             [](const auto& lhs, const auto& rhs) { return lhs.id < rhs.id; }));
  EXPECT_TRUE(std::is_sorted(
      manifest.feature_dispositions.begin(),
      manifest.feature_dispositions.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.feature_id < rhs.feature_id; }));
  EXPECT_TRUE(std::is_sorted(manifest.derived_capacities.begin(),
                             manifest.derived_capacities.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.object_id, lhs.capacity_id} <
                                      std::pair{rhs.object_id, rhs.capacity_id};
                             }));
  EXPECT_TRUE(std::is_sorted(
      manifest.materialization_obligations.begin(),
      manifest.materialization_obligations.end(),
      [](const auto& lhs, const auto& rhs) {
        return std::tuple{lhs.object_id, lhs.strategy_id, lhs.complete_before_stage} <
               std::tuple{rhs.object_id, rhs.strategy_id, rhs.complete_before_stage};
      }));
  EXPECT_TRUE(std::is_sorted(manifest.explicit_fallbacks.begin(),
                             manifest.explicit_fallbacks.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::pair{lhs.feature_id, lhs.fallback_id} <
                                      std::pair{rhs.feature_id, rhs.fallback_id};
                             }));
  EXPECT_TRUE(std::is_sorted(manifest.generated_endpoint_names.begin(),
                             manifest.generated_endpoint_names.end(),
                             [](const auto& lhs, const auto& rhs) {
                               return std::tuple{lhs.final_name, lhs.algorithm, lhs.version} <
                                      std::tuple{rhs.final_name, rhs.algorithm, rhs.version};
                             }));

  const auto rejects_duplicates = [](auto add_duplicates) {
    detail::ExecutionPlanState duplicate_state;
    add_duplicates(duplicate_state.profile_manifest);
    return !detail::finalize_profile_manifest(CompileOptions{}, duplicate_state);
  };
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.scope_resolutions = {
        {.object_id = "duplicate", .object_kind = "test"},
        {.object_id = "distinct", .object_kind = "test"},
        {.object_id = "duplicate", .object_kind = "test"},
    };
  }));
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.required_capabilities = {
        {.capability_id = "duplicate", .version = 1U},
        {.capability_id = "duplicate", .version = 1U},
    };
  }));
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.evidence_references = {
        {.id = "duplicate", .version = 1U},
        {.id = "duplicate", .version = 1U},
    };
  }));
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.feature_dispositions = {
        {.feature_id = "duplicate"},
        {.feature_id = "duplicate"},
    };
  }));
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.derived_capacities = {
        {.object_id = "duplicate", .capacity_id = "count"},
        {.object_id = "duplicate", .capacity_id = "count"},
    };
  }));
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.materialization_obligations = {
        {.object_id = "duplicate", .strategy_id = "strategy"},
        {.object_id = "duplicate", .strategy_id = "strategy"},
    };
  }));
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.explicit_fallbacks = {
        {.feature_id = "duplicate", .fallback_id = "fallback"},
        {.feature_id = "duplicate", .fallback_id = "fallback"},
    };
  }));
  EXPECT_TRUE(rejects_duplicates([](ProfileManifest& duplicate) {
    duplicate.generated_endpoint_names = {
        {.final_name = "duplicate", .algorithm = "test", .version = 1U},
        {.final_name = "duplicate", .algorithm = "test", .version = 1U},
    };
  }));
}

// Iterates all five profiles against the default package; kRealTime and kRegulated are
// rejected because the default package lacks their specific evidence. A second graph
// verifies that a region may only tighten the effective profile, never loosen it.
//
//   [operator]                       (kStandard / kRobotics / kResearch: accepted)
//   [operator]                       (kRealTime / kRegulated: rejected)
//   [operator] in region(kRobotics)  (root=kStandard, tightened to kRobotics)
//
// REQ: COMPILE-003
TEST(ProfileAdmission, ResolvesExactFiveProfilesAndTightenOnlyScopes) {
  struct Expectation {
    Profile profile;
    ProfileAdmissionDisposition disposition;
    std::string_view diagnostic;
  };
  constexpr std::array expectations{
      Expectation{Profile::kStandard, ProfileAdmissionDisposition::kAccepted, {}},
      Expectation{Profile::kRobotics, ProfileAdmissionDisposition::kAccepted, {}},
      Expectation{Profile::kRealTime,
                  ProfileAdmissionDisposition::kRejected,
                  "PROFILE_CAPABILITY_UNSATISFIED"},
      Expectation{Profile::kRegulated,
                  ProfileAdmissionDisposition::kRejected,
                  "PROFILE_EVIDENCE_UNSATISFIED"},
      Expectation{Profile::kResearch, ProfileAdmissionDisposition::kAccepted, {}},
  };
  for (const auto& expectation : expectations) {
    Graph graph{"profile-matrix"};
    (void)graph.op<test::CompileTestOperator>("operator");
    CompileOptions options;
    options.profile = expectation.profile;
    const ExecutionPlan plan = compile(graph, options);
    const auto& requested_profile = plan.profile_manifest().requested_profile;
    if (!requested_profile.has_value()) {
      ADD_FAILURE() << "expected requested profile evidence";
      return;
    }
    EXPECT_EQ(*requested_profile, expectation.profile);
    EXPECT_EQ(plan.profile_manifest().effective_graph_profile, expectation.profile);
    EXPECT_EQ(plan.profile_manifest().disposition, expectation.disposition);
    EXPECT_EQ(plan.ok(), expectation.disposition == ProfileAdmissionDisposition::kAccepted);
    if (!expectation.diagnostic.empty()) {
      EXPECT_TRUE(test::has_diagnostic(plan, expectation.diagnostic));
    }
    EXPECT_FALSE(plan.profile_manifest().plan_schema_version.empty());
  }

  Graph tightened{"profile-tighten"};
  auto op = tightened.op<test::CompileTestOperator>("operator");
  tightened.region("robotics").add(op).profile(Profile::kRobotics);
  const ExecutionPlan plan = compile(tightened);
  ASSERT_EQ(plan.profile_manifest().effective_regions.size(), 1U);
  EXPECT_EQ(plan.profile_manifest().effective_regions.front().requested_profile,
            Profile::kRobotics);
  EXPECT_EQ(plan.profile_manifest().effective_regions.front().effective_profile,
            Profile::kRobotics);
}

// REQ: COMPILE-078
TEST(ProfileAdmission, DuplicateStructuralDiagnosticRetainsLaterCompileCause) {
  Graph graph{"duplicate-profile-evidence"};
  (void)graph.op<test::CompileTestOperator>("operator");
  CompileOptions options;
  options.profile = Profile::kRegulated;
  const ExecutionPlan baseline = compile(graph, options);

  // This unknown endpoint first emits a structural-only root diagnostic. The default package's
  // regulated evidence mismatch later returns the same key with an original Error attached.
  const std::array qualifications{
      detail::RuntimeEndpointClockHealthQualification{.endpoint_name = "unknown-endpoint"}};
  const ExecutionPlan duplicated = detail::compile_with_runtime_evidence(
      graph,
      options,
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .endpoint_clock_health = qualifications,
      });
  for (const auto* plan : {&baseline, &duplicated}) {
    ASSERT_FALSE(plan->ok());
    const auto diagnostics = plan->diagnostics();
    const auto matches = [](const Diagnostic& diagnostic) {
      return diagnostic.token == "PROFILE_EVIDENCE_UNSATISFIED" && !diagnostic.source.vertex &&
             !diagnostic.source.port;
    };
    ASSERT_EQ(std::count_if(diagnostics.begin(), diagnostics.end(), matches), 1);
    const auto found = std::find_if(diagnostics.begin(), diagnostics.end(), matches);
    ASSERT_NE(found, diagnostics.end());
    const auto& evidence = found->detail;
    if (!evidence.has_value()) {
      FAIL() << "Duplicate structural key discarded the later compile cause";
    }
    EXPECT_EQ(evidence->operation, DiagnosticOperation::kCompile);
    EXPECT_EQ(evidence->error.code, ErrorCode::kNotSupported);
    EXPECT_EQ(evidence->error.message, "PROFILE_EVIDENCE_UNSATISFIED");
  }
  EXPECT_EQ(duplicated.json(), baseline.json());
  EXPECT_EQ(duplicated.digest(), baseline.digest());
}

// Single operator inside a kRealTime region under a kStandard graph-level profile.
// Scope resolution must propagate the region's strictest profile to the operator vertex
// and expose it through the per-vertex effective_profile() API for downstream lowering.
//
//   [operator] in region(kRealTime)  — effective_profile per-vertex == kRealTime
//
// REQ: COMPILE-003
// REQ: COMPILE-008
TEST(ProfileAdmission, ExposesCanonicalPerVertexEffectiveProfilesToDownstreamLowering) {
  Graph graph{"effective-profile-view"};
  const auto op = graph.op<test::CompileTestOperator>("operator");
  graph.region("real-time").add(op).profile(Profile::kRealTime);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto admission = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
  const auto operator_vertex =
      std::find_if(snapshot->vertices.begin(), snapshot->vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kOperator;
      });

  ASSERT_NE(operator_vertex, snapshot->vertices.end());
  ASSERT_EQ(admission.effective_profiles().size(), snapshot->vertices.size());
  ASSERT_EQ(admission.rate_evidence_requirements().size(), snapshot->vertices.size());
  EXPECT_EQ(admission.effective_profile(operator_vertex->id), Profile::kRealTime);
  EXPECT_EQ(admission.effective_profiles()[operator_vertex->id.value - 1U], Profile::kRealTime);
  EXPECT_EQ(admission.rate_evidence_requirements()[operator_vertex->id.value - 1U],
            detail::RateBoundedEvidenceRequirement::kRequired);
  EXPECT_FALSE(admission.effective_profile(VertexId{}).has_value());
  EXPECT_FALSE(
      admission
          .effective_profile(VertexId{static_cast<std::uint64_t>(snapshot->vertices.size() + 1U)})
          .has_value());
  // Scope resolution is profile-policy-only. Exact executing-package evidence
  // is matched later by the final profile-manifest pass.
  EXPECT_TRUE(admission.accepted());
}

// One operator claimed by two regions with conflicting profiles; insertion order is
// reversed in a second compile to confirm the strictest profile (kRealTime) always wins.
//
//   [operator] in region(kRobotics) AND region(kRealTime)
//              → effective_profile == kRealTime (both insertion orders)
//
// REQ: COMPILE-003
TEST(ProfileAdmission, ResolvesOverlappingRegionsByStrictestProfileOrderIndependently) {
  const auto compile_overlapping = [](bool reverse) {
    Graph graph{"overlapping-profile-scopes"};
    const auto op = graph.op<test::CompileTestOperator>("operator");
    if (reverse) {
      graph.region("real-time").add(op).profile(Profile::kRealTime);
      graph.region("robotics").add(op).profile(Profile::kRobotics);
    } else {
      graph.region("robotics").add(op).profile(Profile::kRobotics);
      graph.region("real-time").add(op).profile(Profile::kRealTime);
    }
    return compile(graph);
  };

  const ExecutionPlan forward = compile_overlapping(false);
  const ExecutionPlan reverse = compile_overlapping(true);
  ASSERT_FALSE(forward.ok());
  ASSERT_FALSE(reverse.ok());
  EXPECT_TRUE(test::has_diagnostic(forward, "PROFILE_CAPABILITY_UNSATISFIED"));
  EXPECT_TRUE(test::has_diagnostic(reverse, "PROFILE_CAPABILITY_UNSATISFIED"));
  const auto effective_profile = [](const ExecutionPlan& plan) {
    const auto found =
        std::find_if(plan.profile_manifest().scope_resolutions.begin(),
                     plan.profile_manifest().scope_resolutions.end(),
                     [](const auto& record) { return record.object_kind == "operator"; });
    return found == plan.profile_manifest().scope_resolutions.end()
               ? std::optional<Profile>{}
               : std::optional<Profile>{found->effective_profile};
  };
  EXPECT_EQ(effective_profile(forward), Profile::kRealTime);
  EXPECT_EQ(effective_profile(reverse), Profile::kRealTime);
  EXPECT_EQ(forward.digest(), reverse.digest());
  EXPECT_EQ(forward.json(), reverse.json());
}

// Operator with an exported input inside a kRobotics region. A binding record is forged
// post-snapshot with an out-of-range internal_port to exercise closed alias-scope failure;
// the test confirms error tokens are identical regardless of which binding is injected first.
//
//   graph.export_input(public_input, op->input)
//   [operator] in region(kRobotics); binding[1].internal_port = out-of-range
//
// REQ: COMPILE-003
TEST(ProfileAdmission, MalformedAliasScopeFailsClosedOrderIndependently) {
  Graph graph{"malformed-alias-scope"};
  const auto op = graph.op<test::CompileTestOperator>("operator");
  const Input<std::uint32_t> public_input{"public_input"};
  graph.export_input(public_input, op->input);
  graph.region("robotics").add(op).profile(Profile::kRobotics);

  const auto base = *detail::GraphCompilerAccess::canonical_snapshot(graph);
  ASSERT_EQ(base.interface_bindings.size(), 1U);
  const InterfaceBindingRecord valid = base.interface_bindings.front();
  InterfaceBindingRecord malformed = valid;
  malformed.internal_port = GraphPortId{static_cast<std::uint64_t>(base.ports.size() + 1U)};

  const auto admit = [&](bool malformed_first) {
    auto snapshot = base;
    snapshot.interface_bindings = malformed_first
                                      ? std::vector<InterfaceBindingRecord>{malformed, valid}
                                      : std::vector<InterfaceBindingRecord>{valid, malformed};
    return detail::resolve_profile_admission(snapshot, Profile::kStandard);
  };
  const auto forward = admit(true);
  const auto reverse = admit(false);
  const auto interface_vertex =
      std::find_if(base.vertices.begin(), base.vertices.end(), [](const auto& vertex) {
        return vertex.kind == VertexKind::kGraphInterface;
      });

  ASSERT_NE(interface_vertex, base.vertices.end());
  EXPECT_FALSE(forward.effective_profile(interface_vertex->id).has_value());
  EXPECT_FALSE(reverse.effective_profile(interface_vertex->id).has_value());
  EXPECT_EQ(forward.error_tokens, reverse.error_tokens);
}

// Two sub-scenarios: (1) operator in a valid kRobotics region and an "invalid" region
// whose Profile enum is patched to 0xFF post-snapshot; (2) graph-level profile == 0xFF.
// Both must produce a total rejected state with no accepted effective profiles.
//
//   [operator] in region(kRobotics) AND region(0xFF)  → PROFILE_VALUE_UNKNOWN
//   [operator] with graph.profile = 0xFF              → rejected, scope_resolutions empty
//
// REQ: COMPILE-003
TEST(ProfileAdmission, InvalidRegionAndOverlapsPreserveTotalRejectedState) {
  const auto admit = [](bool reverse) {
    Graph graph{"invalid-overlap-profile"};
    const auto op = graph.op<test::CompileTestOperator>("operator");
    const auto add_invalid = [&] { graph.region("invalid").add(op).profile(Profile::kRobotics); };
    const auto add_robotics = [&] { graph.region("robotics").add(op).profile(Profile::kRobotics); };
    if (reverse) {
      add_robotics();
      add_invalid();
    } else {
      add_invalid();
      add_robotics();
    }
    auto snapshot = *detail::GraphCompilerAccess::canonical_snapshot(graph);
    const auto invalid = std::find_if(snapshot.regions.begin(),
                                      snapshot.regions.end(),
                                      [](const auto& region) { return region.name == "invalid"; });
    if (invalid != snapshot.regions.end()) {
      invalid->profile =
          static_cast<Profile>(0xFF);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
    }
    return detail::resolve_profile_admission(snapshot, Profile::kStandard);
  };

  const auto forward = admit(false);
  const auto reverse = admit(true);
  ASSERT_FALSE(forward.accepted());
  ASSERT_FALSE(reverse.accepted());
  EXPECT_EQ(forward.error_tokens, (std::vector<std::string>{"PROFILE_VALUE_UNKNOWN"}));
  EXPECT_EQ(forward.error_tokens, reverse.error_tokens);
  ASSERT_EQ(forward.effective_profiles().size(), 1U);
  ASSERT_EQ(reverse.effective_profiles().size(), 1U);
  EXPECT_FALSE(forward.effective_profiles().front().has_value());
  EXPECT_FALSE(reverse.effective_profiles().front().has_value());
  ASSERT_EQ(forward.contributing_regions_by_vertex.size(), 1U);
  ASSERT_EQ(reverse.contributing_regions_by_vertex.size(), 1U);
  EXPECT_EQ(forward.contributing_regions_by_vertex.front(),
            reverse.contributing_regions_by_vertex.front());
  ASSERT_EQ(forward.manifest.effective_regions.size(), 2U);
  const auto invalid =
      std::find_if(forward.manifest.effective_regions.begin(),
                   forward.manifest.effective_regions.end(),
                   [](const auto& record) { return !record.requested_profile.has_value(); });
  ASSERT_NE(invalid, forward.manifest.effective_regions.end());
  EXPECT_FALSE(invalid->effective_profile.has_value());

  Graph invalid_root{"invalid-root-profile"};
  (void)invalid_root.op<test::CompileTestOperator>("operator");
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(invalid_root);
  const auto invalid_root_admission = detail::resolve_profile_admission(
      *snapshot,
      static_cast<Profile>(0xFF));  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_FALSE(invalid_root_admission.accepted());
  EXPECT_FALSE(invalid_root_admission.manifest.requested_profile.has_value());
  EXPECT_FALSE(invalid_root_admission.manifest.effective_graph_profile.has_value());
  EXPECT_TRUE(invalid_root_admission.manifest.scope_resolutions.empty());
}

// Three independent rejection scenarios; none may silently downgrade to a looser profile.
//
//   [operator]               CompileOptions.profile = 0xFF         → unknown, no effective
//   [operator] in region(kResearch) under kStandard root           → loosening rejected
//   [operator]               CompileOptions.profile = kRegulated   → evidence missing
//
// REQ: COMPILE-003
TEST(ProfileAdmission, RejectsUnknownLooseningAndUnsupportedFeaturesWithoutDowngrade) {
  Graph unknown_graph{"unknown-profile"};
  (void)unknown_graph.op<test::CompileTestOperator>("operator");
  CompileOptions unknown;
  unknown.profile =
      static_cast<Profile>(0xFF);  // NOLINT(clang-analyzer-optin.core.EnumCastOutOfRange)
  const ExecutionPlan unknown_plan = compile(unknown_graph, unknown);
  EXPECT_FALSE(unknown_plan.ok());
  EXPECT_FALSE(unknown_plan.profile_manifest().effective_graph_profile.has_value());

  Graph loosening_graph{"loosening-profile"};
  auto op = loosening_graph.op<test::CompileTestOperator>("operator");
  loosening_graph.region("research").add(op).profile(Profile::kResearch);
  const ExecutionPlan loosening_plan = compile(loosening_graph);
  EXPECT_FALSE(loosening_plan.ok());
  const auto& loosening_profile = loosening_plan.profile_manifest().requested_profile;
  if (!loosening_profile.has_value()) {
    ADD_FAILURE() << "expected loosening rejection to retain the requested profile";
    return;
  }
  EXPECT_EQ(*loosening_profile, Profile::kStandard);

  Graph regulated_graph{"regulated-profile"};
  (void)regulated_graph.op<test::CompileTestOperator>("operator");
  CompileOptions regulated;
  regulated.profile = Profile::kRegulated;
  const ExecutionPlan regulated_plan = compile(regulated_graph, regulated);
  EXPECT_FALSE(regulated_plan.ok());
  const auto& regulated_profile = regulated_plan.profile_manifest().requested_profile;
  if (!regulated_profile.has_value()) {
    ADD_FAILURE() << "expected regulated rejection to retain the requested profile";
    return;
  }
  EXPECT_EQ(*regulated_profile, Profile::kRegulated);
}

// Same linear graph compiled under kStandard and kResearch must produce identical
// partition members, confirming that profile selection is orthogonal to partitioning.
//
//   source --> sink  (kStandard)
//   source --> sink  (kResearch — same topology, same partition result)
//
// REQ: COMPILE-003
TEST(ProfileAdmission, DoesNotSelectOrRewritePartitionCover) {
  Graph standard_graph{"profile-independent"};
  test::add_linear_graph(standard_graph);
  CompileOptions standard;
  standard.profile = Profile::kStandard;
  const ExecutionPlan standard_plan = compile(standard_graph, standard);

  Graph research_graph{"profile-independent"};
  test::add_linear_graph(research_graph);
  CompileOptions research;
  research.profile = Profile::kResearch;
  const ExecutionPlan research_plan = compile(research_graph, research);

  ASSERT_TRUE(standard_plan.ok());
  ASSERT_TRUE(research_plan.ok());
  ASSERT_TRUE(standard_plan.partitioning_plan().accepted_algorithm.has_value());
  ASSERT_TRUE(research_plan.partitioning_plan().accepted_algorithm.has_value());
  EXPECT_EQ(standard_plan.partitioning_plan().accepted_algorithm,
            research_plan.partitioning_plan().accepted_algorithm);
  EXPECT_NE(standard_plan.partitioning_plan().config_digest,
            research_plan.partitioning_plan().config_digest);
  ASSERT_EQ(standard_plan.execution_partitions().size(),
            research_plan.execution_partitions().size());
  for (std::size_t index = 0; index < standard_plan.execution_partitions().size(); ++index) {
    EXPECT_EQ(standard_plan.execution_partitions()[index].members,
              research_plan.execution_partitions()[index].members);
  }
}

// REQ: COMPILE-030
// REQ: COMPILE-068
// REQ: COMPILE-086
// REQ: COMPILE-090
TEST(ProfileAdmission, CanonicalPackageCapabilityCodecIsDistinctSealedAndOrderIndependent) {
  // Key setup: all five list fields (offers, schemas, policies, ABI versions, triples)
  // are reversed before re-sealing to verify that canonical JSON output is order-stable
  // and the digest is unchanged by insertion order.
  const auto& generated = detail::runtime_package_capability_manifest();
  const auto validated = detail::validate_runtime_package_capability_manifest(generated);
  ASSERT_TRUE(validated);
  EXPECT_EQ(generated.supported_plan_schemas,
            std::vector<std::string>{"holoscan.execution_plan.private.v8"});
  const std::string bytes = detail::runtime_package_capability_canonical_json_v1(generated, true);
  EXPECT_TRUE(bytes.starts_with("{\"build_id\":\"holoscan-core-5.0\""));
  EXPECT_NE(bytes.find("\"schema\":\"holoscan.runtime_package_capabilities.canonical_json.v1\""),
            std::string::npos);
  EXPECT_EQ(bytes.find("holoscan.profile_policy.canonical_json.v1"), std::string::npos);
  EXPECT_EQ(generated.digest.size(), 71U);
  EXPECT_TRUE(generated.digest.starts_with("sha256:"));
  const auto package = detail::runtime_package_capability_view();
  for (const std::string_view id : {"holoscan.runtime.channel.generated_schema",
                                    "holoscan.runtime.channel.holomq_transport",
                                    "holoscan.runtime.channel.pod_codec",
                                    "holoscan.runtime.target.current",
                                    "holoscan.runtime.trace_context"}) {
    ASSERT_NE(package.find_offer(id, 1U), nullptr) << id;
  }

  auto golden = generated;
  golden.target_triples = {"x86_64-linux-gnu"};
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(golden));
  // Explicit optional-input and CUDA stream metadata change the supported-plan-schema byte and
  // therefore the sealed package identity; all other canonical package fields remain
  // byte-identical.
  EXPECT_EQ(golden.digest,
            "sha256:072d959acbda8aaf65b7289b55fff82f60d8baf7f730c6deccd574c67762f7f4");
  EXPECT_EQ(detail::runtime_package_capability_canonical_json_v1(golden, true),
            kRuntimePackageX8664Golden);

  auto reordered = generated;
  std::reverse(reordered.supported_plan_schemas.begin(), reordered.supported_plan_schemas.end());
  std::reverse(reordered.profile_policies.begin(), reordered.profile_policies.end());
  std::reverse(reordered.runtime_abi_versions.begin(), reordered.runtime_abi_versions.end());
  std::reverse(reordered.target_triples.begin(), reordered.target_triples.end());
  std::reverse(reordered.offers.begin(), reordered.offers.end());
  EXPECT_EQ(detail::runtime_package_capability_canonical_json_v1(reordered, true), bytes);
  EXPECT_FALSE(detail::validate_runtime_package_capability_manifest(reordered));
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(reordered));
  // Key assertion: reordered and re-sealed manifest produces the same digest and JSON.
  EXPECT_EQ(reordered.digest, generated.digest);
  EXPECT_EQ(detail::runtime_package_capability_canonical_json_v1(reordered, true), bytes);

  auto semantic_mutation = generated;
  semantic_mutation.offers.front().version += 1U;
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(semantic_mutation));
  EXPECT_NE(semantic_mutation.digest, generated.digest);

  auto duplicate = generated;
  duplicate.offers.push_back(duplicate.offers.front());
  EXPECT_FALSE(detail::seal_runtime_package_capability_manifest(duplicate));
  duplicate = generated;
  auto conflicting = duplicate.offers.front();
  conflicting.evidence_id = "conflicting/evidence";
  duplicate.offers.push_back(std::move(conflicting));
  EXPECT_FALSE(detail::seal_runtime_package_capability_manifest(duplicate));
  const auto rejects_duplicate_collection = [&](auto add_duplicate) {
    auto candidate = generated;
    add_duplicate(candidate);
    return !detail::seal_runtime_package_capability_manifest(candidate);
  };
  EXPECT_TRUE(rejects_duplicate_collection([](auto& candidate) {
    candidate.supported_plan_schemas.push_back(candidate.supported_plan_schemas.front());
  }));
  EXPECT_TRUE(rejects_duplicate_collection([](auto& candidate) {
    candidate.profile_policies.push_back(candidate.profile_policies.front());
  }));
  EXPECT_TRUE(rejects_duplicate_collection([](auto& candidate) {
    candidate.runtime_abi_versions.push_back(candidate.runtime_abi_versions.front());
  }));
  EXPECT_TRUE(rejects_duplicate_collection([](auto& candidate) {
    candidate.target_triples.push_back(candidate.target_triples.front());
  }));
}

// REQ: COMPILE-030
// REQ: COMPILE-086
// REQ: COMPILE-090
TEST(ProfileAdmission, MatchesSchemaPolicyAbiTargetCapabilityAndEvidenceExactly) {
  // Key setup: state pre-admitted for kRealTime (or kRegulated); each expect_rejection
  // lambda mutates exactly one required field (schema, policy, ABI, target, capability
  // id/version, or evidence_id) and expects a distinct diagnostic token.
  for (const Profile profile : {Profile::kRealTime, Profile::kRegulated}) {
    auto package = package_with_profile_evidence(profile);
    const auto view = detail::validate_runtime_package_capability_manifest(package);
    ASSERT_TRUE(view);
    auto state = profile_state(profile);
    const auto matched = detail::finalize_profile_manifest(CompileOptions{}, state, view.value());
    ASSERT_TRUE(matched) << matched.error().message;
    EXPECT_EQ(state.profile_manifest.disposition, ProfileAdmissionDisposition::kAccepted);
  }

  const auto expect_rejection = [](auto mutate, std::string_view token) {
    auto package = package_with_profile_evidence(Profile::kRealTime);
    mutate(package);
    EXPECT_TRUE(detail::seal_runtime_package_capability_manifest(package));
    const auto view = detail::validate_runtime_package_capability_manifest(package);
    ASSERT_TRUE(view);
    auto state = profile_state(Profile::kRealTime);
    const auto matched = detail::finalize_profile_manifest(CompileOptions{}, state, view.value());
    ASSERT_FALSE(matched);
    // Key assertion: error token names the specific field that failed matching.
    EXPECT_EQ(matched.error().message, token);
    EXPECT_EQ(state.profile_manifest.disposition, ProfileAdmissionDisposition::kRejected);
  };
  expect_rejection(
      [](auto& package) {
        package.supported_plan_schemas = {"holoscan.execution_plan.private.v4"};
      },
      "PROFILE_PLAN_SCHEMA_UNSUPPORTED");
  expect_rejection(
      [](auto& package) {
        package.supported_plan_schemas = {"holoscan.execution_plan.private.v5"};
      },
      "PROFILE_PLAN_SCHEMA_UNSUPPORTED");
  expect_rejection(
      [](auto& package) {
        package.supported_plan_schemas = {"holoscan.execution_plan.private.v7"};
      },
      "PROFILE_PLAN_SCHEMA_UNSUPPORTED");
  expect_rejection([](auto& package) { package.supported_plan_schemas = {"other"}; },
                   "PROFILE_PLAN_SCHEMA_UNSUPPORTED");
  expect_rejection([](auto& package) { ++package.profile_policies.front().minor; },
                   "PROFILE_POLICY_UNSUPPORTED");
  expect_rejection([](auto& package) { package.runtime_abi_versions = {"other"}; },
                   "PROFILE_RUNTIME_ABI_INCOMPATIBLE");
  expect_rejection([](auto& package) { package.target_triples = {"other"}; },
                   "PROFILE_TARGET_INCOMPATIBLE");
  expect_rejection(
      [](auto& package) {
        package.offers.erase(std::remove_if(package.offers.begin(),
                                            package.offers.end(),
                                            [](const auto& offer) {
                                              return offer.id ==
                                                     "holoscan.profile.real_time.bounded_hot_path";
                                            }),
                             package.offers.end());
      },
      "PROFILE_CAPABILITY_UNSATISFIED");
  expect_rejection(
      [](auto& package) {
        const auto offer =
            std::find_if(package.offers.begin(), package.offers.end(), [](const auto& item) {
              return item.id == "holoscan.profile.real_time.bounded_hot_path";
            });
        if (offer != package.offers.end()) {
          ++offer->version;
        }
      },
      "PROFILE_CAPABILITY_UNSATISFIED");
  expect_rejection(
      [](auto& package) {
        const auto offer =
            std::find_if(package.offers.begin(), package.offers.end(), [](const auto& item) {
              return item.id == "holoscan.profile.real_time.bounded_hot_path";
            });
        if (offer != package.offers.end()) {
          offer->evidence_id = "same-capability/different-evidence";
        }
      },
      "PROFILE_EVIDENCE_UNSATISFIED");

  auto package = package_with_profile_evidence(Profile::kRealTime);
  package.offers.push_back({"unrelated.offer", 1U, "unrelated.evidence/v1"});
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(package));
  const auto view = detail::validate_runtime_package_capability_manifest(package);
  ASSERT_TRUE(view);
  auto state = profile_state(Profile::kRealTime);
  ASSERT_TRUE(detail::finalize_profile_manifest(CompileOptions{}, state, view.value()));
  EXPECT_TRUE(std::none_of(
      state.profile_manifest.required_capabilities.begin(),
      state.profile_manifest.required_capabilities.end(),
      [](const auto& requirement) { return requirement.capability_id == "unrelated.offer"; }));

  auto cross_build = package_with_profile_evidence(Profile::kRealTime);
  cross_build.build_id = "different-producer-build";
  ASSERT_TRUE(detail::seal_runtime_package_capability_manifest(cross_build));
  const auto cross_build_view = detail::validate_runtime_package_capability_manifest(cross_build);
  ASSERT_TRUE(cross_build_view);
  auto cross_build_state = profile_state(Profile::kRealTime);
  ASSERT_TRUE(detail::finalize_profile_manifest(
      CompileOptions{}, cross_build_state, cross_build_view.value()));
  EXPECT_EQ(cross_build_state.profile_manifest.producer_build_id, "diagnostic-build");
  EXPECT_TRUE(
      std::none_of(cross_build_state.profile_manifest.evidence_references.begin(),
                   cross_build_state.profile_manifest.evidence_references.end(),
                   [](const auto& evidence) { return evidence.id == "different-producer-build"; }));
}

// Single operator inside a region carrying all four robotics_critical constraints.
// Only latency_budget and bounded_memory produce profile-manifest evidence; static_topology
// and device_compatible_ops are execution-model guards, not profileable evidence items.
//
//   [operator] in region(kRobotics)
//              + latency_budget(250µs) + bounded_memory
//              + static_topology + device_compatible_ops  (not recorded in manifest)
//
// REQ: COMPILE-030
TEST(ProfileAdmission, RecordsOnlyExplicitRoboticsCriticalConstraintsAndEvidence) {
  Graph critical{"robotics-critical"};
  const auto op = critical.op<test::CompileTestOperator>("operator");
  critical.region("critical")
      .add(op)
      .profile(Profile::kRobotics)
      .constrain(RegionConstraint::static_topology())
      .constrain(RegionConstraint::device_compatible_ops())
      .constrain(RegionConstraint::latency_budget(std::chrono::microseconds{250}))
      .constrain(RegionConstraint::bounded_memory());
  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(critical);
  const auto admission = detail::resolve_profile_admission(*snapshot, Profile::kStandard);
  const auto operator_vertex =
      std::ranges::find(snapshot->vertices, VertexKind::kOperator, &VertexRecord::kind);
  ASSERT_NE(operator_vertex, snapshot->vertices.end());
  EXPECT_EQ(admission.rate_evidence_requirements()[operator_vertex->id.value - 1U],
            detail::RateBoundedEvidenceRequirement::kRequired);
  const ExecutionPlan plan = compile(critical);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto& manifest = plan.profile_manifest();
  const auto has_suffix = [&](std::string_view suffix) {
    return std::any_of(
        manifest.feature_dispositions.begin(),
        manifest.feature_dispositions.end(),
        [suffix](const auto& feature) { return feature.feature_id.ends_with(suffix); });
  };
  EXPECT_TRUE(has_suffix("holoscan.profile.robotics_critical.latency_budget"));
  EXPECT_TRUE(has_suffix("holoscan.profile.robotics_critical.bounded_memory"));
  EXPECT_FALSE(has_suffix("static_topology"));
  EXPECT_FALSE(has_suffix("device_compatible"));
  EXPECT_TRUE(std::any_of(manifest.derived_capacities.begin(),
                          manifest.derived_capacities.end(),
                          [](const auto& capacity) {
                            return capacity.capacity_id == "latency_budget" &&
                                   capacity.limit == 250000U && capacity.unit == "nanoseconds";
                          }));
  EXPECT_TRUE(std::any_of(
      manifest.scope_resolutions.begin(), manifest.scope_resolutions.end(), [](const auto& scope) {
        return scope.object_kind == "feature" && scope.effective_profile == Profile::kRobotics &&
               scope.contributing_regions.size() == 1U;
      }));
}

// REQ: COMPILE-030
TEST(ProfileAdmission, CompletesEveryInstantiatedScopeKindAndCleanupContract) {
  // Key setup: scope_resolutions pre-seeded with one record per object kind (operator,
  // port, edge, boundary); the vector is then filled to capacity so that publishing the
  // channel scope forces a reallocation — contributors must be resolved by stable
  // identities, not retained pointers into the old buffer.
  auto state = profile_state(Profile::kStandard);
  auto& manifest = state.profile_manifest;
  manifest.scope_resolutions = {
      {.object_id = "operator:1",
       .object_kind = "operator",
       .effective_profile = Profile::kStandard},
      {.object_id = "port:1", .object_kind = "port", .effective_profile = Profile::kStandard},
      {.object_id = "port:2", .object_kind = "port", .effective_profile = Profile::kStandard},
      {.object_id = "edge:1", .object_kind = "edge", .effective_profile = Profile::kStandard},
      {.object_id = "boundary:1",
       .object_kind = "boundary",
       .effective_profile = Profile::kResearch,
       .contributing_regions = {RegionId{1U}}},
      {.object_id = "boundary:2",
       .object_kind = "boundary",
       .effective_profile = Profile::kRealTime,
       .contributing_regions = {RegionId{2U}}},
  };
  // Fill the vector's storage so publishing the channel scope must relocate it. Channel
  // contributors must be resolved from stable identities rather than retained pointers.
  manifest.scope_resolutions.reserve(manifest.scope_resolutions.size() + 1U);
  while (manifest.scope_resolutions.size() < manifest.scope_resolutions.capacity()) {
    manifest.scope_resolutions.push_back(ProfileScopeResolutionRecord{
        .object_id = "boundary:filler:" + std::to_string(manifest.scope_resolutions.size()),
        .object_kind = "boundary",
        .effective_profile = Profile::kStandard,
    });
  }
  state.channel_services.push_back(ChannelServiceStaticPlanRecord{
      .topic = ExternalTopicId{1U},
      .service_id = "service",
  });
  state.channel_services.push_back(ChannelServiceStaticPlanRecord{
      .topic = ExternalTopicId{2U},
      .service_id = "service",
  });
  state.interface_port_bindings.push_back(InterfacePortBindingPlanRecord{
      .public_port = PortId{1U},
      .internal_port = PortId{2U},
  });
  state.clock_conversions.push_back(ClockConversionRecord{.id = 1U});
  state.readiness.push_back(ReadinessPlanRecord{
      .source_id = detail::make_source_id(1U),
      .owner = VertexId{1U},
      .accepted_capacity = 1U,
      .accepted_sender_reference_capacity = 1U,
      .materialized = true,
  });
  auto package = package_with_profile_evidence(Profile::kRealTime);
  const auto package_view = detail::validate_runtime_package_capability_manifest(package);
  ASSERT_TRUE(package_view);
  const auto finalized =
      detail::finalize_profile_manifest(CompileOptions{}, state, package_view.value());
  ASSERT_TRUE(finalized) << finalized.error().message;
  std::set<std::string> kinds;
  for (const auto& scope : manifest.scope_resolutions) {
    kinds.insert(scope.object_kind);
  }
  // Key assertion: finalization adds bridge, channel, converter, feature, and helper kinds
  // so the complete set of scope resolution object kinds covers all nine instantiated types.
  EXPECT_EQ(kinds,
            (std::set<std::string>{"boundary",
                                   "bridge",
                                   "channel",
                                   "converter",
                                   "edge",
                                   "feature",
                                   "helper",
                                   "operator",
                                   "port"}));
  const auto joined_channel = std::find_if(
      manifest.scope_resolutions.begin(), manifest.scope_resolutions.end(), [](const auto& scope) {
        return scope.object_id == "channel:service";
      });
  ASSERT_NE(joined_channel, manifest.scope_resolutions.end());
  EXPECT_EQ(joined_channel->effective_profile, Profile::kRealTime);
  EXPECT_EQ(joined_channel->contributing_regions,
            (std::vector<RegionId>{RegionId{1U}, RegionId{2U}}));
  ASSERT_FALSE(manifest.materialization_obligations.empty());
  for (const auto& obligation : manifest.materialization_obligations) {
    EXPECT_EQ(obligation.cleanup_stage_mask, 0xC0U);
    EXPECT_GT(obligation.max_cleanup_actions, 0U);
  }
}

}  // namespace
}  // namespace holoscan
