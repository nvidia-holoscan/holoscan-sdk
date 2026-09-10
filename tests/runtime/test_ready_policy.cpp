/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/execution_plan.hpp>

#include "../common/allocation_counter.hpp"
#include "core/source_id_internal.hpp"
#include "runtime/scheduling/deterministic_serial_ready_policy.hpp"
#include "runtime/scheduling/ready_policy_catalog.hpp"

namespace holoscan {
namespace {

using detail::make_source_id;
using detail::scheduling::DeterministicSerialReadyPolicy;
using detail::scheduling::ReadyCandidateView;
using detail::scheduling::ReadyPolicyCatalog;
using detail::scheduling::ReadyPolicyRegistration;
using detail::scheduling::ReadyPolicyState;
using detail::scheduling::ReadySetView;
using detail::scheduling::SchedulerPlanView;

template <typename T>
concept HasTagMember = requires(T value) { value.tag; };

// REQ: RUNTIME-002
static_assert(!HasTagMember<ReadyCandidateView>);

// Returns the minimal single-lane serial plan record used by most policy initialization tests.
[[nodiscard]] SchedulerPlanRecord valid_scheduler_plan() {
  return SchedulerPlanRecord{
      .ready_policy = detail::scheduling::deterministic_serial_ready_policy_algorithm(),
      .execution_lane_count = 1U,
      .max_concurrent_activations = 1U,
      .activation_overlap = false,
      .worker_count = 1U,
      .intra_path_parallelism = 1U,
      .reason = {},
  };
}

// Wraps a SchedulerPlanRecord in a SchedulerPlanView with a configurable candidate cap.
// Used alongside valid_scheduler_plan() to drive DeterministicSerialReadyPolicy::initialize.
[[nodiscard]] SchedulerPlanView scheduler_view(const SchedulerPlanRecord& plan,
                                               std::size_t max_ready_candidates = 16U) {
  return SchedulerPlanView{
      .scheduler = &plan,
      .max_ready_candidates = max_ready_candidates,
      .available_capabilities = detail::scheduling::kAllReadyPolicyCapabilities,
  };
}

// Builds a ReadyCandidateView with all six tie-break fields individually settable.
// Default values are all 1 so callers vary only the field under test; used by
// UsesStableTotalTieBreak and PartitionAndVertexAreEvidenceNotPriority.
[[nodiscard]] ReadyCandidateView candidate(std::int64_t logical_ns = 100,
                                           std::uint32_t microstep = 0U,
                                           std::uint32_t plan_vertex_order = 1U,
                                           std::uint32_t source = 1U, std::uint64_t port = 1U,
                                           std::uint64_t tie = 1U, std::uint32_t partition = 1U,
                                           std::uint64_t vertex = 1U) {
  return ReadyCandidateView{
      .partition = ExecutionPartitionId{partition},
      .vertex = VertexId{vertex},
      .tag_time = TagTime{logical_ns, microstep, 0U},
      .plan_vertex_order = plan_vertex_order,
      .stable_source_id = make_source_id(source),
      .stable_port_id = PortId{port},
      .stable_candidate_tie = tie,
  };
}

// Asserts that a catalog result carries the expected ErrorCode and diagnostic token string.
void expect_error(const expected<ReadyPolicyCatalog, Error>& result, ErrorCode code,
                  std::string_view token) {
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, code);
  EXPECT_EQ(result.error().message, token);
}

// The catalog is a cold constructor-injected table. The baseline's built-in table contains only the
// production deterministic-serial implementation; the future topology-aware identifier is not a
// registration or fallback.
// REQ: RUNTIME-002
TEST(ReadyPolicyCatalogTest, RegistersOnlyDeterministicSerial) {
  auto catalog = ReadyPolicyCatalog::create_default_builtins();
  ASSERT_TRUE(catalog);
  EXPECT_EQ(catalog->size(), 1U);

  auto policy = catalog->make(detail::scheduling::deterministic_serial_ready_policy_algorithm());
  ASSERT_TRUE(policy);
  const auto descriptor = policy.value()->descriptor();
  EXPECT_EQ(descriptor.algorithm.id, "holoscan.ready.deterministic_serial");
  EXPECT_EQ(descriptor.algorithm.version, 1U);
  EXPECT_EQ(descriptor.interface_version, 1U);
  EXPECT_EQ(descriptor.maturity, AlgorithmMaturity::kProduction);

  auto topology = catalog->make(AlgorithmKey{"holoscan.ready.topology_aware", 1U});
  ASSERT_FALSE(topology);
  EXPECT_EQ(topology.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(topology.error().message, "READY_POLICY_NOT_FOUND");
}

// REQ: RUNTIME-002
TEST(ReadyPolicyCatalogTest, RejectsUnknownVersionAndMalformedRegistrationExactly) {
  auto catalog = ReadyPolicyCatalog::create_default_builtins();
  ASSERT_TRUE(catalog);

  auto unknown = catalog->make(AlgorithmKey{"holoscan.ready.unknown", 1U});
  ASSERT_FALSE(unknown);
  EXPECT_EQ(unknown.error().code, ErrorCode::kNotFound);
  EXPECT_EQ(unknown.error().message, "READY_POLICY_NOT_FOUND");

  auto version = catalog->make(AlgorithmKey{"holoscan.ready.deterministic_serial", 2U});
  ASSERT_FALSE(version);
  EXPECT_EQ(version.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(version.error().message, "READY_POLICY_VERSION_UNSUPPORTED");

  expect_error(ReadyPolicyCatalog::create({}),
               ErrorCode::kInvalidArgument,
               "READY_POLICY_REGISTRATION_INVALID");

  const DeterministicSerialReadyPolicy implementation;
  const ReadyPolicyRegistration valid{
      implementation.descriptor(),
      &detail::scheduling::make_deterministic_serial_ready_policy,
  };
  const std::array duplicate{valid, valid};
  expect_error(ReadyPolicyCatalog::create(duplicate),
               ErrorCode::kInvalidArgument,
               "READY_POLICY_REGISTRATION_INVALID");

  auto unsupported_interface = valid;
  unsupported_interface.descriptor.interface_version += 1U;
  const std::array unsupported_interface_table{unsupported_interface};
  expect_error(ReadyPolicyCatalog::create(unsupported_interface_table),
               ErrorCode::kNotSupported,
               "READY_POLICY_INTERFACE_VERSION_UNSUPPORTED");

  const std::array valid_table{valid};
  expect_error(ReadyPolicyCatalog::create(valid_table, {}),
               ErrorCode::kNotSupported,
               "READY_POLICY_PLAN_INVALID");
}

// REQ: RUNTIME-002
TEST(DeterministicSerialReadyPolicyTest, UsesStableTotalTieBreak) {
  DeterministicSerialReadyPolicy policy;
  ReadyPolicyState state;
  const auto plan = valid_scheduler_plan();
  ASSERT_TRUE(policy.initialize(scheduler_view(plan), state));

  // Key setup: each pair places the higher-priority item at index 1 (later=index 0);
  //            fields are varied one level at a time in priority order:
  //            ns → microstep → plan_vertex_order → source → port → tie.
  const auto expect_first = [&](ReadyCandidateView expected, ReadyCandidateView later) {
    const std::array ready{later, expected};
    const auto selected = policy.select(ReadySetView{ready}, state);
    ASSERT_TRUE(selected);
    // Key assertion: selected.value == 1U confirms the lower field value dominates in every tier.
    EXPECT_EQ(selected.value, 1U);
  };

  expect_first(candidate(99, 9U, 99U, 99U, 99U, 99U), candidate(100, 0U, 0U, 0U, 0U, 0U));
  expect_first(candidate(100, 0U, 99U, 99U, 99U, 99U), candidate(100, 1U, 0U, 0U, 0U, 0U));
  expect_first(candidate(100, 0U, 1U, 99U, 99U, 99U), candidate(100, 0U, 2U, 0U, 0U, 0U));
  expect_first(candidate(100, 0U, 1U, 1U, 99U, 99U), candidate(100, 0U, 1U, 2U, 0U, 0U));
  expect_first(candidate(100, 0U, 1U, 1U, 1U, 99U), candidate(100, 0U, 1U, 1U, 2U, 0U));
  expect_first(candidate(100, 0U, 1U, 1U, 1U, 1U), candidate(100, 0U, 1U, 1U, 1U, 2U));
}

// REQ: RUNTIME-002
TEST(DeterministicSerialReadyPolicyTest, PartitionAndVertexAreEvidenceNotPriority) {
  DeterministicSerialReadyPolicy policy;
  ReadyPolicyState state;
  const auto plan = valid_scheduler_plan();
  ASSERT_TRUE(policy.initialize(scheduler_view(plan), state));

  // Key setup: candidate[1] has tie=1 (wins) but partition=99 and vertex=99,
  //            confirming those identity fields play no role in priority ordering.
  const std::array ready{
      candidate(100, 0U, 1U, 1U, 1U, 2U, 1U, 1U),
      candidate(100, 0U, 1U, 1U, 1U, 1U, 99U, 99U),
  };
  const auto selected = policy.select(ReadySetView{ready}, state);
  ASSERT_TRUE(selected);
  // Key assertion: selected.value == 1U despite candidate[1] carrying the largest identity IDs.
  EXPECT_EQ(selected.value, 1U);
}

// REQ: RUNTIME-002
TEST(DeterministicSerialReadyPolicyTest, RejectsInvalidOrIncompatiblePlanState) {
  DeterministicSerialReadyPolicy policy;

  // Key setup: each invalid condition is isolated — zero candidate cap, wrong policy version,
  //            zero lanes, multiple partition-local workers, max_concurrent_activations > 1,
  //            activation_overlap,
  //            and empty capabilities — each must yield READY_POLICY_PLAN_INVALID.
  const auto expect_plan_invalid = [&](const SchedulerPlanRecord& plan,
                                       std::size_t max_ready_candidates,
                                       detail::scheduling::ReadyPolicyCapabilities capabilities) {
    ReadyPolicyState state;
    auto view = scheduler_view(plan, max_ready_candidates);
    view.available_capabilities = capabilities;
    auto result = policy.initialize(view, state);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
    EXPECT_EQ(result.error().message, "READY_POLICY_PLAN_INVALID");
    EXPECT_FALSE(state.initialized);
  };

  auto plan = valid_scheduler_plan();
  expect_plan_invalid(plan, 0U, detail::scheduling::kAllReadyPolicyCapabilities);

  plan = valid_scheduler_plan();
  plan.ready_policy.version = 2U;
  expect_plan_invalid(plan, 1U, detail::scheduling::kAllReadyPolicyCapabilities);

  plan = valid_scheduler_plan();
  plan.execution_lane_count = 0U;
  expect_plan_invalid(plan, 1U, detail::scheduling::kAllReadyPolicyCapabilities);

  plan = valid_scheduler_plan();
  plan.worker_count = 2U;
  expect_plan_invalid(plan, 1U, detail::scheduling::kAllReadyPolicyCapabilities);

  plan = valid_scheduler_plan();
  plan.max_concurrent_activations = 2U;
  expect_plan_invalid(plan, 1U, detail::scheduling::kAllReadyPolicyCapabilities);

  plan = valid_scheduler_plan();
  plan.activation_overlap = true;
  expect_plan_invalid(plan, 1U, detail::scheduling::kAllReadyPolicyCapabilities);

  plan = valid_scheduler_plan();
  expect_plan_invalid(plan, 1U, {});
}

// REQ: RUNTIME-002
// REQ: RUNTIME-006
TEST(DeterministicSerialReadyPolicyTest, InitializesOnePartitionLocalPolicyWithinAMultiLanePlan) {
  DeterministicSerialReadyPolicy policy;
  ReadyPolicyState state;
  auto plan = valid_scheduler_plan();
  plan.execution_lane_count = 3U;
  plan.worker_count = 1U;
  // Key setup: the complete plan has three placed lanes, while this policy instance owns one
  // partition-local serial interval and therefore receives overlap disabled.
  // Key assertion: the partition-local state initializes without flattening the plan lane count.
  EXPECT_TRUE(policy.initialize(scheduler_view(plan), state));
  EXPECT_TRUE(state.initialized);
  EXPECT_FALSE(state.activation_in_progress);
}

// The policy object and its fixed-size state survive catalog destruction. The
// immutable scheduler view is consumed once during initialization; no planner,
// proposal, authoring graph, or catalog is retained by the selection path.
// REQ: RUNTIME-002
TEST(ReadyPolicyLifetimeTest, SelectsOnceAndRetainsNoCompilerObjects) {
  static_assert(std::is_trivially_copyable_v<SchedulerPlanView>);
  static_assert(std::is_trivially_copyable_v<ReadyPolicyState>);

  // Key setup: the catalog is moved out of scope and destroyed before initialize and select;
  //            only the policy object and a stack-local SchedulerPlanView survive.
  const auto plan = valid_scheduler_plan();
  std::unique_ptr<detail::scheduling::ReadyPolicy> policy;
  {
    auto catalog = ReadyPolicyCatalog::create_default_builtins();
    ASSERT_TRUE(catalog);
    auto created = catalog->make(plan.ready_policy);
    ASSERT_TRUE(created);
    policy = std::move(*created);
  }

  ReadyPolicyState state;
  ASSERT_TRUE(policy->initialize(scheduler_view(plan, 2U), state));
  auto second_initialize = policy->initialize(scheduler_view(plan, 2U), state);
  ASSERT_FALSE(second_initialize);
  EXPECT_EQ(second_initialize.error().message, "READY_POLICY_PLAN_INVALID");

  const std::array ready{candidate(101), candidate(100)};
  const auto selected = policy->select(ReadySetView{ready}, state);
  ASSERT_TRUE(selected);
  // Key assertion: index 1 (logical_ns=100) wins over index 0 (logical_ns=101), confirming
  //               the policy selects correctly without retaining any catalog or plan data.
  EXPECT_EQ(selected.value, 1U);
}

// One partition-local policy instance admits one activation at a time. Other
// partitions own independent policy state and may overlap this interval.
// REQ: RUNTIME-002
// REQ: RUNTIME-011
TEST(DeterministicSerialReadyPolicyTest, SerializesOnlyItsOwnedPartitionActivation) {
  DeterministicSerialReadyPolicy policy;
  ReadyPolicyState state;
  auto plan = valid_scheduler_plan();
  plan.execution_lane_count = 3U;
  plan.worker_count = 1U;
  ASSERT_TRUE(policy.initialize(scheduler_view(plan, 3U), state));

  // Key setup: three candidates are presented to this partition-local policy instance.
  const std::array ready{
      candidate(100, 0U, 1U, 1U, 1U, 1U, 1U, 1U),
      candidate(100, 0U, 2U, 2U, 2U, 2U, 2U, 2U),
      candidate(100, 0U, 3U, 3U, 3U, 3U, 3U, 3U),
  };
  const auto first = policy.select(ReadySetView{ready}, state);
  ASSERT_TRUE(first);
  policy.on_activation_started(ready[first.value], state);
  // Key assertion: this policy instance returns no second candidate until its owned activation
  // completes; independent partition policy instances are not represented by this state.
  EXPECT_FALSE(policy.select(ReadySetView{ready}, state));
  policy.on_activation_completed(ready[first.value], state);
  EXPECT_TRUE(policy.select(ReadySetView{ready}, state));
}

// REQ: RUNTIME-002
TEST(DeterministicSerialReadyPolicyTest, UsesOnlyPreinitializedBoundedState) {
  static_assert(std::is_trivially_copyable_v<ReadyPolicyState>);

  DeterministicSerialReadyPolicy policy;
  ReadyPolicyState state;
  // Key setup: max_ready_candidates=2; the too_many array has 3 entries which exceeds the cap.
  const auto plan = valid_scheduler_plan();
  ASSERT_TRUE(policy.initialize(scheduler_view(plan, 2U), state));
  EXPECT_TRUE(state.initialized);
  EXPECT_EQ(state.max_ready_candidates, 2U);
  EXPECT_FALSE(state.activation_in_progress);

  const auto selected = candidate();
  policy.on_activation_started(selected, state);
  EXPECT_TRUE(state.activation_in_progress);
  policy.on_activation_completed(selected, state);
  EXPECT_FALSE(state.activation_in_progress);

  const std::array too_many{candidate(), candidate(101), candidate(102)};
  // Key assertion: selecting from an oversized or empty set returns nullopt without state change.
  EXPECT_FALSE(policy.select(ReadySetView{too_many}, state));
  EXPECT_FALSE(policy.select(ReadySetView{}, state));

  auto reinitialize = policy.initialize(scheduler_view(plan, 2U), state);
  ASSERT_FALSE(reinitialize);
  EXPECT_EQ(reinitialize.error().message, "READY_POLICY_PLAN_INVALID");
}

// REQ: RUNTIME-002
TEST(ReadyPolicyHotPathTest, NoAllocationTraversalOrLookupAfterWarmup) {
  DeterministicSerialReadyPolicy policy;
  ReadyPolicyState state;
  const auto plan = valid_scheduler_plan();
  ASSERT_TRUE(policy.initialize(scheduler_view(plan, 3U), state));
  const std::array ready{candidate(101), candidate(100), candidate(102)};
  // Key setup: one warmup selection runs before the counter is installed to prime any
  //            internal state that could trigger a first-time allocation.
  ASSERT_TRUE(policy.select(ReadySetView{ready}, state));

  std::size_t allocations = 0U;
  bool selected_every_time = true;
  {
    const test::ScopedAllocationCounter counter;
    for (std::size_t iteration = 0U; iteration < 128U; ++iteration) {
      const auto selected = policy.select(ReadySetView{ready}, state);
      selected_every_time = selected_every_time && selected && selected.value == 1U;
      policy.on_activation_started(ready[selected.value], state);
      policy.on_activation_completed(ready[selected.value], state);
    }
    allocations = counter.count();
  }
  EXPECT_TRUE(selected_every_time);
  // Key assertion: allocations == 0 over 128 select/on_activation_started/completed cycles.
  EXPECT_EQ(allocations, 0U);
}

}  // namespace
}  // namespace holoscan
