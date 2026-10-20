/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <vector>

#include <holoscan/core/compile.hpp>

#include "compiler/partitioning/partition_canonicalization.hpp"
#include "compiler/partitioning/partition_planner_catalog.hpp"
#include "compiler/partitioning/partitioning_pass.hpp"
#include "partition_test_helpers.hpp"

namespace holoscan {
namespace {

using detail::partitioning::CandidatePartitionId;
using detail::partitioning::ExecutionDomainKey;
using detail::partitioning::PartitionBoundary;
using detail::partitioning::PartitioningPass;
using detail::partitioning::PartitionPlanner;
using detail::partitioning::PartitionPlannerCatalog;
using detail::partitioning::PartitionPlannerDescriptor;
using detail::partitioning::PartitionPlannerRegistration;
using detail::partitioning::PartitionProblem;
using detail::partitioning::PartitionProposal;
using detail::partitioning::PartitionSelectionInput;
using detail::partitioning::ValidatedPartitionPlan;

// Every mandatory boundary bit maps to one stable, canonical reason sequence.
// Used to keep serial planning, source-chain planning, and validation on one authority.
TEST(PartitionCanonicalizationTest, ProjectsEveryBoundaryReasonInCanonicalOrder) {
  // Key setup: add every mandatory boundary in deliberately noncanonical order.
  detail::partitioning::PartitionBoundaryMask mask;
  mask.add(PartitionBoundary::kReadiness);
  mask.add(PartitionBoundary::kLifecycle);
  mask.add(PartitionBoundary::kExplicitThreadPlacement);
  mask.add(PartitionBoundary::kRegionBackend);
  mask.add(PartitionBoundary::kDeviceTransfer);
  mask.add(PartitionBoundary::kExternalEndpoint);
  mask.add(PartitionBoundary::kProcess);

  std::vector<PartitionDecisionReason> reasons;
  detail::partitioning::append_boundary_reasons(mask, reasons);

  // Key assertion: projection follows the one stable boundary-reason authority.
  EXPECT_EQ(reasons,
            (std::vector{PartitionDecisionReason::kProcessBoundary,
                         PartitionDecisionReason::kExternalEndpointBoundary,
                         PartitionDecisionReason::kDeviceTransferIncompatibility,
                         PartitionDecisionReason::kRegionBackendAdmission,
                         PartitionDecisionReason::kExplicitThreadPlacement,
                         PartitionDecisionReason::kLifecycleIsolation,
                         PartitionDecisionReason::kReadinessIsolation}));
}

// Returns the canonical partition ID assigned to vertex_id, or 0 if the vertex is absent.
// Used throughout topology-assertion tests to compare partition membership.
[[nodiscard]] std::uint32_t partition_of(const ValidatedPartitionPlan& plan,
                                         std::uint64_t vertex_id) {
  const auto found = std::find_if(plan.decisions.begin(), plan.decisions.end(), [&](const auto& d) {
    return d.vertex.value == vertex_id;
  });
  return found == plan.decisions.end() ? 0U : found->canonical_partition_id;
}

// Asserts that lhs and rhs produce identical partition membership and decision lists;
// used in determinism tests to confirm permutation-invariant output.
void expect_same_cover(const ValidatedPartitionPlan& lhs, const ValidatedPartitionPlan& rhs) {
  ASSERT_EQ(lhs.partitions.size(), rhs.partitions.size());
  ASSERT_EQ(lhs.decisions.size(), rhs.decisions.size());
  for (std::size_t index = 0; index < lhs.partitions.size(); ++index) {
    EXPECT_EQ(lhs.partitions[index].canonical_id, rhs.partitions[index].canonical_id);
    EXPECT_EQ(lhs.partitions[index].members, rhs.partitions[index].members);
  }
  for (std::size_t index = 0; index < lhs.decisions.size(); ++index) {
    EXPECT_EQ(lhs.decisions[index].vertex, rhs.decisions[index].vertex);
    EXPECT_EQ(lhs.decisions[index].canonical_partition_id,
              rhs.decisions[index].canonical_partition_id);
    EXPECT_EQ(lhs.decisions[index].decision, rhs.decisions[index].decision);
    EXPECT_EQ(lhs.decisions[index].reasons, rhs.decisions[index].reasons);
  }
}

// A source-chain planner stub that always returns an empty (invalid) proposal.
// Used to verify the PartitioningPass rejects bad proposals without silently falling back
// (e.g. InvalidExperimentalProposalDoesNotFallback, FailedPlanRetainsAttemptProvenance).
class InvalidSourceChainPlanner final : public PartitionPlanner {
 public:
  [[nodiscard]] PartitionPlannerDescriptor descriptor() const noexcept override {
    return {detail::partitioning::source_chain_algorithm(),
            detail::partitioning::kPartitionPlannerInterfaceVersion,
            AlgorithmMaturity::kProduction,
            detail::partitioning::kAllPartitionPlannerCapabilities};
  }

  [[nodiscard]] expected<PartitionProposal, Error> propose(const PartitionProblem&) const override {
    return PartitionProposal{detail::partitioning::source_chain_algorithm(), {}, {}};
  }
};

[[nodiscard]] std::unique_ptr<PartitionPlanner> make_invalid_source_chain() {
  return std::make_unique<InvalidSourceChainPlanner>();
}

// Compile-time surface probes used in NoPublicSelectorOrPrivateHeaderLeak and
// PlannerProblemWithholdsProfilesAndWholeGraphIdentity; each concept detects a field
// that must NOT be present in the probed type.
template <typename T>
concept HasPartitionSelector = requires(T value) { value.partition_algorithm; };

template <typename T>
concept HasProfileInput = requires(T value) { value.profile; };

template <typename T>
concept HasWholeGraphHash = requires(T value) { value.graph_hash; };

template <typename T>
concept HasAuthoringGeneration = requires(T value) { value.generation; };

// REQ: COMPILE-004
// REQ: COMPILE-009
// REQ: COMPILE-095
TEST(PartitioningPrivacyTest, PrivateHeadersAndNeutralNames) {
  EXPECT_EQ(detail::partitioning::kSerialBaselineAlgorithmId, "holoscan.partition.serial_baseline");
  EXPECT_EQ(detail::partitioning::kSourceChainAlgorithmId, "holoscan.partition.source_chain");
  EXPECT_EQ(detail::partitioning::kPartitionPlannerInterfaceVersion, 3U);
  EXPECT_EQ(detail::partitioning::kPartitionAlgorithmVersion, 1U);
  EXPECT_EQ(detail::partitioning::partition_environment_variable(),
            "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM");
}

// REQ: COMPILE-003
// REQ: COMPILE-006
TEST(PartitioningPrivacyTest, PlannerProblemWithholdsProfilesAndWholeGraphIdentity) {
  static_assert(!HasProfileInput<PartitionProblem>);
  static_assert(!HasWholeGraphHash<PartitionProblem>);
  static_assert(!HasAuthoringGeneration<PartitionProblem>);
  const test::PartitionProblemStorage storage{1U};
  EXPECT_EQ(storage.problem().vertices.size(), 1U);
}

// REQ: COMPILE-004
TEST(PartitionPlannerCatalogTest, RegistersExactBuiltins) {
  auto catalog = PartitionPlannerCatalog::create_default_builtins();
  ASSERT_TRUE(catalog);
  auto baseline = catalog->make(detail::partitioning::serial_baseline_algorithm());
  auto source_chain = catalog->make(detail::partitioning::source_chain_algorithm());
  ASSERT_TRUE(baseline);
  ASSERT_TRUE(source_chain);
  EXPECT_EQ(baseline.value()->descriptor().maturity, AlgorithmMaturity::kProduction);
  EXPECT_EQ(source_chain.value()->descriptor().maturity, AlgorithmMaturity::kProduction);
  EXPECT_FALSE(catalog->make(AlgorithmKey{"holoscan.partition.unknown", 1U}));
}

// REQ: COMPILE-068
TEST(PartitionPlannerSelectorTest, OrdinaryCompileUsesSourceChain) {
  auto catalog = PartitionPlannerCatalog::create_default_builtins();
  ASSERT_TRUE(catalog);
  const test::PartitionProblemStorage storage{2U, {{0U, 1U}}};
  const PartitionSelectionInput selection{};
  auto result = PartitioningPass(*catalog).run(selection, storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->requested_algorithm, detail::partitioning::source_chain_algorithm());
  EXPECT_EQ(result->selection_source, PlanSelectionSource::kCoreDefault);
  EXPECT_EQ(result->maturity, AlgorithmMaturity::kProduction);
  EXPECT_FALSE(result->experimental_warning_required);
}

// Serial-baseline merges v1→v2 (same domain) but splits at v3 (different execution domain).
//
//   v1 --> v2 --> v3[domain=2]
//
// REQ: COMPILE-004
TEST(SerialBaselinePartitionPlannerTest, ProducesCoarsestLegalDomainCover) {
  test::PartitionProblemStorage storage{3U, {{0U, 1U}, {1U, 2U}}};
  storage.vertices[2].execution_domain = detail::partitioning::ExecutionDomainKey{2U};
  auto result = test::run_serial_baseline(storage.problem());
  ASSERT_TRUE(result);
  ASSERT_EQ(result->partitions.size(), 2U);
  // Key assertion: v1 and v2 share a partition; v3 is in a separate partition (domain=2).
  EXPECT_EQ(partition_of(*result, 1U), partition_of(*result, 2U));
  EXPECT_NE(partition_of(*result, 2U), partition_of(*result, 3U));
}

// REQ: COMPILE-004
// REQ: COMPILE-009
// REQ: COMPILE-095
TEST(PartitioningPublicSurfaceTest, NoPublicSelectorOrPrivateHeaderLeak) {
  static_assert(!HasPartitionSelector<CompileOptions>);
  EXPECT_FALSE((std::is_constructible_v<PartitioningPass, CompileOptions>));
}

// REQ: COMPILE-068
TEST(PartitionPlannerSelectorTest, RecordsInternalSourceChainSelection) {
  auto catalog = PartitionPlannerCatalog::create_default_builtins();
  ASSERT_TRUE(catalog);
  const test::PartitionProblemStorage storage{2U, {{0U, 1U}}};
  const auto requested = detail::partitioning::source_chain_algorithm();
  const PartitionSelectionInput selection{requested};
  auto result = PartitioningPass(*catalog).run(selection, storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->requested_algorithm, detail::partitioning::source_chain_algorithm());
  EXPECT_EQ(result->selection_source, PlanSelectionSource::kCompilerService);
  EXPECT_FALSE(result->experimental_warning_required);
}

// A single vertex and a fully-connected linear chain each collapse to one partition.
//
//   single: v1
//   chain:  v1 --> v2 --> v3 --> v4
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, SingleVertexAndLinearChain) {
  const test::PartitionProblemStorage single{1U};
  auto one = test::run_source_chain(single.problem());
  ASSERT_TRUE(one);
  EXPECT_EQ(one->partitions.size(), 1U);

  const test::PartitionProblemStorage linear{4U, {{0U, 1U}, {1U, 2U}, {2U, 3U}}};
  auto chain = test::run_source_chain(linear.problem());
  ASSERT_TRUE(chain);
  EXPECT_EQ(chain->partitions.size(), 1U);
}

// Disconnected components each start their own chain; two partitions result.
//
//   v1 --> v2    v3 --> v4
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, MultipleRootsAndDisconnectedComponents) {
  const test::PartitionProblemStorage storage{4U, {{0U, 1U}, {2U, 3U}}};
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 2U);
  EXPECT_EQ(partition_of(*result, 1U), partition_of(*result, 2U));
  EXPECT_EQ(partition_of(*result, 3U), partition_of(*result, 4U));
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 3U));
}

// A fan-in target must start its own partition regardless of predecessor count.
//
//   v1 --> v3 <-- v2   (v3 is placed in a third partition)
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, FanInUsesDistinctPredecessorPartitions) {
  const test::PartitionProblemStorage storage{3U, {{0U, 2U}, {1U, 2U}}};
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 3U);
  EXPECT_NE(partition_of(*result, 3U), partition_of(*result, 1U));
  EXPECT_NE(partition_of(*result, 3U), partition_of(*result, 2U));
}

// Each fan-out leg starts its own partition; three partitions result.
//
//   v1 --> v2
//      \-> v3
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, FanOutCreatesLegPartitions) {
  const test::PartitionProblemStorage storage{3U, {{0U, 1U}, {0U, 2U}}};
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 3U);
}

// Source, two fan-out legs, and the re-convergence each occupy their own partition.
//
//   v1 --> v2 --> v4
//      \-> v3 ->/
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, DiamondSplitsAndReconverges) {
  const test::PartitionProblemStorage storage{4U, {{0U, 1U}, {0U, 2U}, {1U, 3U}, {2U, 3U}}};
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 4U);
}

// Parallel edges to the same successor are not structural fan-out; one partition results.
//
//   v1 ===> v2   (two edges, single successor)
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, ParallelEdgesToSameSuccessorAreNotFanOut) {
  const test::PartitionProblemStorage storage{2U, {{0U, 1U}, {0U, 1U}}};
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 1U);
}

// A shortcut edge (v1→v3) alongside v1→v2→v3 makes this a literal fan-out + fan-in.
//
//   v1 --> v2 --> v3
//      \------->/
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, ShortcutEdgeStillCreatesLiteralFanOutAndFanIn) {
  const test::PartitionProblemStorage storage{3U, {{0U, 1U}, {0U, 2U}, {1U, 2U}}};
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  ASSERT_EQ(result->partitions.size(), 3U);
  // Key assertion: v2 is FanOutLegSplit; v3 is FanInSplit (shortcut forces both roles).
  EXPECT_EQ(result->decisions[1].decision, PartitionDecisionKind::kFanOutLegSplit);
  EXPECT_EQ(result->decisions[1].reasons.front(), PartitionDecisionReason::kFanOutLeg);
  EXPECT_EQ(result->decisions[2].decision, PartitionDecisionKind::kFanInSplit);
  EXPECT_EQ(result->decisions[2].reasons.front(), PartitionDecisionReason::kFanIn);
}

// starts_execution_chain forces v2 into its own partition even in a simple linear chain.
//
//   v1 --> v2[starts_execution_chain]
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, IndependentReadinessStartsChain) {
  test::PartitionProblemStorage storage{2U, {{0U, 1U}}};
  storage.vertices[1].starts_execution_chain = true;
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 2U);
  EXPECT_EQ(result->decisions[1].reasons.front(),
            PartitionDecisionReason::kIndependentReadinessRoot);
}

// Mandatory boundaries before v2 (kProcess) and v3 (kExternalEndpoint) force three splits.
//
//   v1 --> |kProcess| v2 --> |kExternalEndpoint| v3
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, PreservesProcessAndExternalBoundaries) {
  test::PartitionProblemStorage storage{3U, {{0U, 1U}, {1U, 2U}}};
  storage.vertices[1].boundary_before.add(PartitionBoundary::kProcess);
  storage.vertices[2].boundary_before.add(PartitionBoundary::kExternalEndpoint);
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 3U);
}

// Four consecutive mandatory boundaries produce five partitions in a linear chain.
//
//   v1 --[device]--> v2 --[region]--> v3 --[thread]--> v4 --[lifecycle]--> v5
//
// REQ: COMPILE-006
TEST(SourceChainPartitionPlannerTest, PreservesDeviceRegionRoleAndLifecycleBoundaries) {
  test::PartitionProblemStorage storage{5U, {{0U, 1U}, {1U, 2U}, {2U, 3U}, {3U, 4U}}};
  storage.vertices[1].boundary_before.add(PartitionBoundary::kDeviceTransfer);
  storage.vertices[2].boundary_before.add(PartitionBoundary::kRegionBackend);
  storage.vertices[3].boundary_before.add(PartitionBoundary::kExplicitThreadPlacement);
  storage.vertices[4].boundary_before.add(PartitionBoundary::kLifecycle);
  auto result = test::run_source_chain(storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->partitions.size(), 5U);
}

// Rejoining a serialization domain after fan-out would make the partition quotient cyclic, so the
// terminal domain member is split from the source.
//
//   v1[d] --> v2 --\
//      \----> v3 ----> v4[d]
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, SerializationDomainSplitPreservesAcyclicQuotient) {
  test::PartitionProblemStorage storage{4U, {{0U, 1U}, {0U, 2U}, {1U, 3U}, {2U, 3U}}};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[3].serialization_domain = detail::partitioning::SerializationDomainKey{1U};

  const auto result = test::run_source_chain(storage.problem());

  // Key assertion: the legal split is accepted without contracting the diamond into a cycle.
  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(result->partitions.size(), 4U);
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 4U));
  EXPECT_EQ(result->decisions[3].decision, PartitionDecisionKind::kSerializationDomain);
}

// A mandatory boundary remains authoritative between connected members of one serialization
// domain.
//
//   v1[d] --> |readiness| v2[d]
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, SerializationDomainSplitPreservesMandatoryBoundary) {
  test::PartitionProblemStorage storage{2U, {{0U, 1U}}};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[1].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[1].boundary_before.add(PartitionBoundary::kReadiness);

  const auto result = test::run_source_chain(storage.problem());

  // Key assertion: the serialization request cannot erase the readiness boundary.
  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(result->partitions.size(), 2U);
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 2U));
}

// Incompatible execution domains remain separate even when disconnected Operators request the
// same serialization domain.
//
//   v1[d, execution=1]   v2[d, execution=2]
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, SerializationDomainSplitPreservesExecutionDomain) {
  test::PartitionProblemStorage storage{2U};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[1].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[1].execution_domain = detail::partitioning::ExecutionDomainKey{2U};

  const auto result = test::run_source_chain(storage.problem());

  // Key assertion: serialization cannot merge incompatible execution domains.
  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(result->partitions.size(), 2U);
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 2U));
}

// A later compatible member reuses the first legal split for its serialization domain.
//
//   v1[d, execution=1]   v2[d, execution=2]   v3[d, execution=2]
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, SerializationDomainReusesCompatibleSplitCandidate) {
  test::PartitionProblemStorage storage{3U};
  for (auto& vertex : storage.vertices) {
    vertex.serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  }
  storage.vertices[1].execution_domain = detail::partitioning::ExecutionDomainKey{2U};
  storage.vertices[2].execution_domain = detail::partitioning::ExecutionDomainKey{2U};

  const auto result = test::run_source_chain(storage.problem());

  // Key assertion: the second legal candidate is reused instead of creating a third partition.
  ASSERT_TRUE(result) << result.error().message;
  ASSERT_EQ(result->partitions.size(), 2U);
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 2U));
  EXPECT_EQ(partition_of(*result, 2U), partition_of(*result, 3U));
}

// A serialization-domain member does not exempt unrelated plain members from chain validation.
//
//   v1[d]   v2   (an invalid disconnected mixed partition)
//
// REQ: COMPILE-093
TEST(PartitionPlanValidatorTest, RejectsDisconnectedMixedSerializationPartition) {
  test::PartitionProblemStorage storage{2U};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  const detail::partitioning::SourceChainPartitionPlanner planner;
  auto proposal = planner.propose(storage.problem());
  ASSERT_TRUE(proposal);
  ASSERT_EQ(proposal->assignments.size(), 2U);
  ASSERT_EQ(proposal->decisions.size(), 2U);
  proposal->assignments[1].partition = proposal->assignments[0].partition;
  proposal->decisions[1].partition = proposal->decisions[0].partition;

  const auto result = detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), *proposal);

  // Key assertion: mixed partitions retain ordinary connectivity validation.
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_COVER_INVALID");
}

// A native tail makes the first serialization partition structural, so a later disconnected root
// starts another same-domain candidate instead of invalidating the generated cover.
//
//   v1[d] -> v2       v3[d]
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, NativeTailBeforeAnotherSerializedRoot) {
  test::PartitionProblemStorage storage{3U, {{0U, 1U}}};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[2].serialization_domain = detail::partitioning::SerializationDomainKey{1U};

  const auto result = test::run_source_chain(storage.problem());

  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(partition_of(*result, 1U), partition_of(*result, 2U));
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 3U));
}

// A disconnected serialized root may share a pure serialization partition, but its presence keeps
// a later native tail from turning that partition into an invalid disconnected mixed cover.
//
//   v1[d] -> v3       v2[d]
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, NativeTailAfterAnotherSerializedRoot) {
  test::PartitionProblemStorage storage{3U, {{0U, 2U}}};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[1].serialization_domain = detail::partitioning::SerializationDomainKey{1U};

  const auto result = test::run_source_chain(storage.problem());

  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(partition_of(*result, 1U), partition_of(*result, 2U));
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 3U));
}

// Two ordinary serialized roots with native tails remain admissible without placing disconnected
// native work in one mixed partition.
//
//   v1[d] -> v2       v3[d] -> v4
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, TwoSerializedRootsWithNativeTails) {
  test::PartitionProblemStorage storage{4U, {{0U, 1U}, {2U, 3U}}};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[2].serialization_domain = detail::partitioning::SerializationDomainKey{1U};

  const auto result = test::run_source_chain(storage.problem());

  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(partition_of(*result, 1U), partition_of(*result, 2U));
  EXPECT_EQ(partition_of(*result, 3U), partition_of(*result, 4U));
  EXPECT_NE(partition_of(*result, 1U), partition_of(*result, 3U));
}

// Disconnected compatible members still share their canonical serialization partition.
//
//   v1[d]   v2[d]
//
// REQ: COMPILE-093
TEST(SourceChainPartitionPlannerTest, SerializationDomainRejectsAnUnnecessarySplit) {
  test::PartitionProblemStorage storage{2U};
  storage.vertices[0].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  storage.vertices[1].serialization_domain = detail::partitioning::SerializationDomainKey{1U};
  const detail::partitioning::SourceChainPartitionPlanner planner;
  auto proposal = planner.propose(storage.problem());
  ASSERT_TRUE(proposal);
  proposal->assignments[1].partition = CandidatePartitionId{2U};
  proposal->decisions[1].partition = CandidatePartitionId{2U};

  const auto result = detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), *proposal);

  // Key assertion: validator independence prevents a gratuitous same-domain split.
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_COVER_INVALID");
}

// Canonical IDs must be stable regardless of the internal ordering of proposal assignments.
//
//   v1 --> v2 --> v3
//
// REQ: COMPILE-006
TEST(PartitioningPassTest, EmbeddedGraphIdentityIsCanonical) {
  const test::PartitionProblemStorage storage{3U, {{0U, 1U}, {1U, 2U}}};
  const detail::partitioning::SourceChainPartitionPlanner planner;
  auto proposal = planner.propose(storage.problem());
  ASSERT_TRUE(proposal);
  auto first = detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), *proposal);
  // Key setup: assignments and decisions are reversed to test ordering invariance.
  std::reverse(proposal->assignments.begin(), proposal->assignments.end());
  std::reverse(proposal->decisions.begin(), proposal->decisions.end());
  auto reordered = detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), *proposal);
  ASSERT_TRUE(first);
  ASSERT_TRUE(reordered);
  // Key assertion: reversed proposal produces an identical canonical partition cover.
  expect_same_cover(*first, *reordered);
}

// Diamond compiled with edges in forward vs. reverse insertion order must yield the same cover.
//
//   v1 --> v2 --> v4
//      \-> v3 ->/
//
// REQ: COMPILE-006
TEST(PartitioningDeterminismTest, OperatorInsertionPermutation) {
  const test::PartitionProblemStorage first_storage{4U, {{0U, 1U}, {0U, 2U}, {1U, 3U}, {2U, 3U}}};
  const test::PartitionProblemStorage second_storage{4U, {{2U, 3U}, {1U, 3U}, {0U, 2U}, {0U, 1U}}};
  auto first = test::run_source_chain(first_storage.problem());
  auto second = test::run_source_chain(second_storage.problem());
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  expect_same_cover(*first, *second);
}

// Shortcut topology compiled with two different edge orderings must produce identical covers.
//
//   v1 --> v2 --> v3
//      \------->/
//
// REQ: COMPILE-006
TEST(PartitioningDeterminismTest, EdgeOrderHashSeedAndAddressIndependence) {
  const test::PartitionProblemStorage first_storage{3U, {{0U, 1U}, {1U, 2U}, {0U, 2U}}};
  const test::PartitionProblemStorage second_storage{3U, {{0U, 2U}, {1U, 2U}, {0U, 1U}}};
  auto first = test::run_source_chain(first_storage.problem());
  auto second = test::run_source_chain(second_storage.problem());
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  expect_same_cover(*first, *second);
}

// Compiling the same diamond topology twice must produce byte-identical partition plans.
//
//   v1 --> v2 --> v4
//      \-> v3 ->/
//
// REQ: COMPILE-006
TEST(PartitioningDeterminismTest, RepeatedCompileIsByteIdentical) {
  const test::PartitionProblemStorage storage{4U, {{0U, 1U}, {0U, 2U}, {1U, 3U}, {2U, 3U}}};
  auto first = test::run_source_chain(storage.problem());
  auto second = test::run_source_chain(storage.problem());
  ASSERT_TRUE(first);
  ASSERT_TRUE(second);
  expect_same_cover(*first, *second);
}

// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, RejectsAlgorithmIdentityMismatch) {
  const test::PartitionProblemStorage storage{1U};
  const detail::partitioning::SerialBaselinePartitionPlanner planner;
  auto proposal = planner.propose(storage.problem());
  ASSERT_TRUE(proposal);
  proposal->algorithm = detail::partitioning::source_chain_algorithm();
  auto result = detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), *proposal);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_PROPOSAL_ALGORITHM_MISMATCH");
}

// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, RejectsUnknownMissingDuplicateAndEmptyAssignments) {
  const test::PartitionProblemStorage storage{2U, {{0U, 1U}}};
  const detail::partitioning::SerialBaselinePartitionPlanner planner;
  auto proposal = planner.propose(storage.problem());
  ASSERT_TRUE(proposal);

  auto missing = *proposal;
  missing.assignments.pop_back();
  EXPECT_FALSE(detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), missing));

  auto duplicate = *proposal;
  duplicate.assignments[1].vertex = duplicate.assignments[0].vertex;
  EXPECT_FALSE(detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), duplicate));

  auto empty = *proposal;
  empty.assignments[0].partition = CandidatePartitionId{};
  EXPECT_FALSE(detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), empty));

  auto unknown = *proposal;
  unknown.assignments[0].vertex = 99U;
  EXPECT_FALSE(detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), unknown));
}

// A mandatory kProcess boundary on the edge and same_partition_collapse_legal=false are
// each rejected when the serial-baseline collapses the two-vertex graph.
//
//   v1 --> |kProcess| v2   (baseline collapse crosses the mandatory boundary)
//
// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, RejectsBoundaryCrossingAndIllegalCollapse) {
  const detail::partitioning::SerialBaselinePartitionPlanner planner;
  // Key setup: mandatory kProcess boundary makes the baseline's collapse proposal invalid.
  test::PartitionProblemStorage boundary{2U, {{0U, 1U}}};
  boundary.edges[0].mandatory_boundaries.add(PartitionBoundary::kProcess);
  auto boundary_proposal = planner.propose(boundary.problem());
  ASSERT_TRUE(boundary_proposal);
  // Key assertion: proposal that crosses a mandatory boundary is rejected.
  EXPECT_FALSE(detail::partitioning::validate_partition_proposal(
      boundary.problem(), planner.descriptor(), *boundary_proposal));

  test::PartitionProblemStorage illegal{2U, {{0U, 1U}}};
  illegal.edges[0].same_partition_collapse_legal = false;
  auto illegal_proposal = planner.propose(illegal.problem());
  ASSERT_TRUE(illegal_proposal);
  EXPECT_FALSE(detail::partitioning::validate_partition_proposal(
      illegal.problem(), planner.descriptor(), *illegal_proposal));
}

// Two sub-cases: cross-domain collapse and a gratuitous split on identical-domain vertices.
//
//   cross-domain: v1 --> v2[domain=2]  (forced join across domains is illegal)
//   gratuitous:   v1     v2            (no edges; baseline need not split them)
//
// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, RejectsCrossDomainCollapseAndNonCoarsestBaselineSplit) {
  const detail::partitioning::SerialBaselinePartitionPlanner planner;

  test::PartitionProblemStorage cross_domain{2U};
  cross_domain.vertices[1].execution_domain = ExecutionDomainKey{2U};
  auto cross_domain_proposal = planner.propose(cross_domain.problem());
  ASSERT_TRUE(cross_domain_proposal);
  // Key setup: proposal forces v2 into v1's partition despite the domain mismatch.
  cross_domain_proposal->assignments[1].partition = cross_domain_proposal->assignments[0].partition;
  cross_domain_proposal->decisions[1].partition = cross_domain_proposal->assignments[0].partition;
  cross_domain_proposal->decisions[1].decision = PartitionDecisionKind::kJoin;
  auto collapsed = detail::partitioning::validate_partition_proposal(
      cross_domain.problem(), planner.descriptor(), *cross_domain_proposal);
  // Key assertion: cross-domain collapse is rejected with PARTITION_COLLAPSE_ILLEGAL.
  ASSERT_FALSE(collapsed);
  EXPECT_EQ(collapsed.error().message, "PARTITION_COLLAPSE_ILLEGAL");

  const test::PartitionProblemStorage gratuitous_split{2U};
  auto split_proposal = planner.propose(gratuitous_split.problem());
  ASSERT_TRUE(split_proposal);
  split_proposal->assignments[1].partition = CandidatePartitionId{2U};
  split_proposal->decisions[1].partition = CandidatePartitionId{2U};
  split_proposal->decisions[1].decision = PartitionDecisionKind::kStart;
  auto split = detail::partitioning::validate_partition_proposal(
      gratuitous_split.problem(), planner.descriptor(), *split_proposal);
  ASSERT_FALSE(split);
  EXPECT_EQ(split.error().message, "PARTITION_COVER_INVALID");
}

// A source-chain proposal must not use self-justifying reasons to claim a partition
// decision that the graph topology does not support.
//
// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, RejectsSourceChainProposalThatSelfJustifiesIllegalAssignment) {
  const detail::partitioning::SourceChainPartitionPlanner planner;

  const test::PartitionProblemStorage legal_join{2U, {{0U, 1U}}};
  auto split_proposal = planner.propose(legal_join.problem());
  ASSERT_TRUE(split_proposal);
  // Key setup: proposal splits a legally-joinable edge using a self-justifying reason.
  split_proposal->assignments[1].partition = CandidatePartitionId{2U};
  split_proposal->decisions[1].partition = CandidatePartitionId{2U};
  split_proposal->decisions[1].decision = PartitionDecisionKind::kCollapseRejected;
  split_proposal->decisions[1].reasons = {PartitionDecisionReason::kCollapseRejected};
  auto split = detail::partitioning::validate_partition_proposal(
      legal_join.problem(), planner.descriptor(), *split_proposal);
  // Key assertion: self-justifying CollapseRejected reason is not accepted as evidence.
  ASSERT_FALSE(split);
  EXPECT_EQ(split.error().message, "PARTITION_COVER_INVALID");

  const test::PartitionProblemStorage unrelated_join{3U, {{0U, 2U}}};
  auto join_proposal = planner.propose(unrelated_join.problem());
  ASSERT_TRUE(join_proposal);
  join_proposal->assignments[2].partition = join_proposal->assignments[1].partition;
  join_proposal->decisions[2].partition = join_proposal->assignments[1].partition;
  join_proposal->decisions[2].decision = PartitionDecisionKind::kJoin;
  join_proposal->decisions[2].reasons = {PartitionDecisionReason::kLinearChainJoin};
  auto joined = detail::partitioning::validate_partition_proposal(
      unrelated_join.problem(), planner.descriptor(), *join_proposal);
  ASSERT_FALSE(joined);
  EXPECT_EQ(joined.error().message, "PARTITION_COVER_INVALID");

  const test::PartitionProblemStorage independent_roots{2U};
  auto roots_proposal = planner.propose(independent_roots.problem());
  ASSERT_TRUE(roots_proposal);
  roots_proposal->assignments[1].partition = roots_proposal->assignments[0].partition;
  roots_proposal->decisions[1].partition = roots_proposal->assignments[0].partition;
  auto roots = detail::partitioning::validate_partition_proposal(
      independent_roots.problem(), planner.descriptor(), *roots_proposal);
  ASSERT_FALSE(roots);
  EXPECT_EQ(roots.error().message, "PARTITION_COVER_INVALID");
}

// A hand-crafted proposal that creates a cycle in the partition quotient graph must be rejected.
//
//   v1 --> v2 --> v3   (proposal: partition{1}→{2}→{1}, a cycle)
//
// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, RejectsCyclicQuotientGraph) {
  const test::PartitionProblemStorage storage{3U, {{0U, 1U}, {1U, 2U}}};
  const auto descriptor = detail::partitioning::SerialBaselinePartitionPlanner{}.descriptor();
  const PartitionProposal proposal{
      detail::partitioning::serial_baseline_algorithm(),
      {{0U, {1U}}, {1U, {2U}}, {2U, {1U}}},
      {{0U, {1U}, PartitionDecisionKind::kStart, {}},
       {1U, {2U}, PartitionDecisionKind::kStart, {}},
       {2U, {1U}, PartitionDecisionKind::kJoin, {}}},
  };
  auto result =
      detail::partitioning::validate_partition_proposal(storage.problem(), descriptor, proposal);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_QUOTIENT_CYCLE");
}

// REQ: COMPILE-068
TEST(PartitionPlanValidatorTest, RejectsPlannerGrowthOrSplitOfExactAnchor) {
  const test::PartitionProblemStorage storage{3U, {{0U, 1U}, {1U, 2U}}};
  const std::array anchor_members{1U, 2U};
  const std::array constraints{detail::partitioning::AuthoredPartitionConstraintView{
      detail::partitioning::AuthoredPartitionConstraintId{7U}, anchor_members}};
  auto problem = storage.problem();
  problem.authored_constraints = constraints;
  const detail::partitioning::SourceChainPartitionPlanner planner;
  auto proposal = planner.propose(problem);
  ASSERT_TRUE(proposal);

  const CandidatePartitionId anchor_candidate = proposal->assignments[1].partition;
  auto grown = *proposal;
  grown.assignments[0].partition = anchor_candidate;
  grown.decisions[0].partition = anchor_candidate;
  auto grown_result =
      detail::partitioning::validate_partition_proposal(problem, planner.descriptor(), grown);
  ASSERT_FALSE(grown_result);
  EXPECT_EQ(grown_result.error().message, "PARTITION_PROPOSAL_CONSTRAINT_VIOLATION");

  auto split = *proposal;
  split.assignments[2].partition = CandidatePartitionId{99U};
  split.decisions[2].partition = CandidatePartitionId{99U};
  auto split_result =
      detail::partitioning::validate_partition_proposal(problem, planner.descriptor(), split);
  ASSERT_FALSE(split_result);
  EXPECT_EQ(split_result.error().message, "PARTITION_PROPOSAL_CONSTRAINT_VIOLATION");
}

// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, EnforcesPartitionLimit) {
  const test::PartitionProblemStorage storage{2U};
  const auto descriptor = detail::partitioning::SerialBaselinePartitionPlanner{}.descriptor();
  const PartitionProposal proposal{
      detail::partitioning::serial_baseline_algorithm(),
      {{0U, {1U}}, {1U, {2U}}},
      {{0U, {1U}, PartitionDecisionKind::kStart, {}},
       {1U, {2U}, PartitionDecisionKind::kStart, {}}},
  };
  auto result =
      detail::partitioning::validate_partition_proposal(storage.problem(1U), descriptor, proposal);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_LIMIT_EXCEEDED");
}

// REQ: COMPILE-006
TEST(PartitionPlanValidatorTest, ValidatesAndOrdersDecisionReasons) {
  test::PartitionProblemStorage storage{1U};
  storage.vertices[0].boundary_before.add(PartitionBoundary::kProcess);
  storage.vertices[0].boundary_before.add(PartitionBoundary::kExternalEndpoint);
  const detail::partitioning::SerialBaselinePartitionPlanner planner;
  auto proposal = planner.propose(storage.problem());
  ASSERT_TRUE(proposal);
  std::reverse(proposal->decisions[0].reasons.begin(), proposal->decisions[0].reasons.end());
  auto result = detail::partitioning::validate_partition_proposal(
      storage.problem(), planner.descriptor(), *proposal);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_DECISION_REASON_INVALID");
}

// Non-sequential proposal IDs (99, 7) must be remapped to canonical IDs (1, 2) in
// topological order; v1 (domain=2) triggers a domain-split from v2 and v3.
//
//   v1[domain=2] --> v2 --> v3   (v2 and v3 share partition 2)
//
// REQ: COMPILE-006
TEST(PartitionCanonicalizerTest, AssignsStableIdsAndPreservesScheduleEdges) {
  test::PartitionProblemStorage storage{3U, {{0U, 1U}, {1U, 2U}}};
  storage.vertices[0].execution_domain = ExecutionDomainKey{2U};
  const auto descriptor = detail::partitioning::SerialBaselinePartitionPlanner{}.descriptor();
  // Key setup: proposal uses non-sequential IDs (99, 7); validator must canonicalize them.
  const PartitionProposal proposal{
      detail::partitioning::serial_baseline_algorithm(),
      {{0U, {99U}}, {1U, {7U}}, {2U, {7U}}},
      {{0U, {99U}, PartitionDecisionKind::kStart, {}},
       {1U, {7U}, PartitionDecisionKind::kStart, {}},
       {2U, {7U}, PartitionDecisionKind::kJoin, {}}},
  };
  auto result =
      detail::partitioning::validate_partition_proposal(storage.problem(), descriptor, proposal);
  ASSERT_TRUE(result);
  ASSERT_EQ(result->partitions.size(), 2U);
  // Key assertion: canonical IDs are 1 and 2 in topological order regardless of proposal IDs.
  EXPECT_EQ(result->partitions[0].canonical_id, 1U);
  EXPECT_EQ(result->partitions[0].members, std::vector<VertexId>({VertexId{1U}}));
  EXPECT_EQ(result->partitions[1].canonical_id, 2U);
  EXPECT_EQ(result->partitions[1].members, std::vector<VertexId>({VertexId{2U}, VertexId{3U}}));
}

// REQ: COMPILE-005
// REQ: COMPILE-006
TEST(PartitioningPassTest, InvalidExperimentalProposalDoesNotFallback) {
  const PartitionPlannerRegistration registration{InvalidSourceChainPlanner{}.descriptor(),
                                                  &make_invalid_source_chain};
  auto catalog = PartitionPlannerCatalog::create(std::span{&registration, 1U});
  ASSERT_TRUE(catalog);
  const test::PartitionProblemStorage storage{1U};
  const auto source_chain = detail::partitioning::source_chain_algorithm();
  const PartitionSelectionInput selection{std::nullopt, source_chain, Profile::kStandard, {}};
  auto result = PartitioningPass(*catalog).run(selection, storage.problem());
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().message, "PARTITION_COVER_INVALID");
}

// REQ: COMPILE-005
// REQ: COMPILE-006
TEST(PartitioningPassTest, FailedPlanRetainsAttemptProvenance) {
  const PartitionPlannerRegistration registration{InvalidSourceChainPlanner{}.descriptor(),
                                                  &make_invalid_source_chain};
  auto catalog = PartitionPlannerCatalog::create(std::span{&registration, 1U});
  ASSERT_TRUE(catalog);
  const test::PartitionProblemStorage storage{1U};
  const auto requested = detail::partitioning::source_chain_algorithm();
  const PartitionSelectionInput selection{requested, std::nullopt, Profile::kStandard, {}};
  auto result = PartitioningPass(*catalog).run(selection, storage.problem());
  EXPECT_FALSE(result);
  ASSERT_TRUE(selection.compiler_service_request.has_value());
  EXPECT_EQ(*selection.compiler_service_request, requested);
}

// The shadow proposal from the serial baseline must not appear as the accepted algorithm
// evidence when the source-chain planner is explicitly requested.
//
//   v1 --> v2
//      \-> v3   (fan-out; source chain produces 3 partitions)
//
// REQ: COMPILE-006
TEST(PartitioningPassTest, ShadowProposalIsNotAcceptedPlanEvidence) {
  auto catalog = PartitionPlannerCatalog::create_default_builtins();
  ASSERT_TRUE(catalog);
  const test::PartitionProblemStorage storage{3U, {{0U, 1U}, {0U, 2U}}};
  const auto requested = detail::partitioning::source_chain_algorithm();
  const PartitionSelectionInput selection{requested, std::nullopt, Profile::kStandard, {}};
  auto result = PartitioningPass(*catalog).run(selection, storage.problem());
  ASSERT_TRUE(result);
  EXPECT_EQ(result->plan.algorithm, requested);
  EXPECT_NE(result->plan.algorithm, detail::partitioning::serial_baseline_algorithm());
}

}  // namespace
}  // namespace holoscan
