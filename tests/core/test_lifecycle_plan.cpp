/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/lifecycle_kernel.hpp"

namespace holoscan::test {
namespace {

using detail::LifecycleKernelCapacity;
using detail::LifecycleKernelErrorCode;
using detail::LifecycleNodeId;
using detail::LifecycleStageGraphInput;

// Derives the LifecycleMethodId for a stage: the convention is that method IDs are
// stage ordinal + 1 so each stage maps to exactly one callable method slot.
[[nodiscard]] constexpr LifecycleMethodId method_for(LifecycleStage stage) noexcept {
  return static_cast<LifecycleMethodId>(static_cast<std::uint8_t>(stage) + 1U);
}

// Sizes a LifecycleKernelCapacity to fit exactly the given input, with 6× headroom
// for completion records (one slot per stage transition per node).
[[nodiscard]] LifecycleKernelCapacity capacity_for(const LifecycleStageGraphInput& input) {
  return LifecycleKernelCapacity{
      .max_scopes = input.scopes.size(),
      .max_nodes = input.nodes.size(),
      .max_callbacks = input.callbacks.size(),
      .max_dependency_records = input.producers.size() + input.requirements.size(),
      .max_direct_edges = input.direct_edges.size(),
      .max_completions = input.nodes.size() * 6U,
  };
}

// REQ: LIFECYCLE-003
// Graph under test (kConfigure stage):
//
//   Scope 7
//   [first:10] --[configured fact]--> [second:20]
//   [first:10] --[direct edge]------> [third:30]
//
//   kConfigure levels: {10} then {20, 30}
//   kDiscover  levels: {10, 20, 30}  (no ordering constraints)
//
// Representative: verifies sparse callbacks (only "third" has one), producer/consumer
// fact linking, and that the same graph rejects cycles and misconfigured dependencies.
TEST(LifecycleDependencyPlan, BuildsStableSparseDagsAndRejectsCycles) {
  // Key setup: three nodes in scope 7; only node 30 has a callback (sparse coverage).
  LifecycleStageGraphInput input;
  input.scopes = {{.scope_id = 7U}};
  input.nodes = {
      {.node_id = 30U, .scope_id = 7U, .stable_name = "third", .stable_order = 3U},
      {.node_id = 10U, .scope_id = 7U, .stable_name = "first", .stable_order = 1U},
      {.node_id = 20U, .scope_id = 7U, .stable_name = "second", .stable_order = 2U},
  };
  // The producer has no callback. Its successful logical no-op checkpoint still satisfies the
  // ordering fact, while the one registered callback demonstrates sparse callback metadata.
  input.callbacks = {{.node_id = 30U,
                      .stage = LifecycleStage::kStart,
                      .method = method_for(LifecycleStage::kStart)}};
  input.producers = {
      {.node_id = 10U,
       .declared_stage = LifecycleStage::kConfigure,
       .key = {.scope_id = 7U, .stage = LifecycleStage::kConfigure, .stable_name = "configured"}}};
  input.requirements = {
      {.node_id = 20U,
       .declared_stage = LifecycleStage::kConfigure,
       .key = {.scope_id = 7U, .stage = LifecycleStage::kConfigure, .stable_name = "configured"}}};
  input.direct_edges = {{.declaration_scope_id = 7U,
                         .before_node_id = 10U,
                         .before_stage = LifecycleStage::kConfigure,
                         .after_node_id = 30U,
                         .after_stage = LifecycleStage::kConfigure}};

  // Key assertion: kConfigure is two levels; kDiscover collapses to one because it has
  // no ordering constraints of its own.
  const auto graph = detail::build_lifecycle_stage_graph(input, capacity_for(input));
  ASSERT_TRUE(graph.has_value());
  const auto& configure = graph->stages[static_cast<std::size_t>(LifecycleStage::kConfigure)];
  ASSERT_EQ(configure.levels.size(), 2U);
  EXPECT_EQ(configure.levels[0].node_ids, std::vector<LifecycleNodeId>({10U}));
  EXPECT_EQ(configure.levels[1].node_ids, std::vector<LifecycleNodeId>({20U, 30U}));

  const auto& discover = graph->stages[static_cast<std::size_t>(LifecycleStage::kDiscover)];
  ASSERT_EQ(discover.levels.size(), 1U);
  EXPECT_EQ(discover.levels[0].node_ids, std::vector<LifecycleNodeId>({10U, 20U, 30U}));

  auto cycle = input;
  cycle.direct_edges.push_back({.declaration_scope_id = 7U,
                                .before_node_id = 20U,
                                .before_stage = LifecycleStage::kConfigure,
                                .after_node_id = 10U,
                                .after_stage = LifecycleStage::kConfigure});
  auto rejected = detail::build_lifecycle_stage_graph(cycle, capacity_for(cycle));
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, LifecycleKernelErrorCode::kCycle);

  auto missing = input;
  missing.producers.clear();
  rejected = detail::build_lifecycle_stage_graph(missing, capacity_for(missing));
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, LifecycleKernelErrorCode::kMissingProducer);

  auto duplicate = input;
  duplicate.producers.push_back(
      {.node_id = 30U,
       .declared_stage = LifecycleStage::kConfigure,
       .key = {.scope_id = 7U, .stage = LifecycleStage::kConfigure, .stable_name = "configured"}});
  rejected = detail::build_lifecycle_stage_graph(duplicate, capacity_for(duplicate));
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, LifecycleKernelErrorCode::kDuplicateProducer);

  auto cross_stage = input;
  cross_stage.requirements[0].declared_stage = LifecycleStage::kDiscover;
  rejected = detail::build_lifecycle_stage_graph(cross_stage, capacity_for(cross_stage));
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, LifecycleKernelErrorCode::kCrossStageDependency);

  auto cross_scope = input;
  cross_scope.requirements[0].key.scope_id = 99U;
  rejected = detail::build_lifecycle_stage_graph(cross_scope, capacity_for(cross_scope));
  ASSERT_FALSE(rejected.has_value());
  EXPECT_EQ(rejected.error().code, LifecycleKernelErrorCode::kCrossScopeDependency);
}

// REQ: LIFECYCLE-002
// REQ: LIFECYCLE-003
// Graph under test (well-formed baseline):
//
//   Scope 7
//   [first:1] --[kConfigure direct edge]--> [second:2]
//
// Each sub-case mutates one field of the baseline and expects a specific error code.
TEST(LifecycleDependencyPlan, RejectsMalformedScopesNodesCallbacksAndCrossStageEdges) {
  LifecycleStageGraphInput input;
  input.scopes = {{.scope_id = 7U}};
  input.nodes = {
      {.node_id = 1U, .scope_id = 7U, .stable_name = "first", .stable_order = 1U},
      {.node_id = 2U, .scope_id = 7U, .stable_name = "second", .stable_order = 2U},
  };
  input.callbacks = {{.node_id = 1U,
                      .stage = LifecycleStage::kStart,
                      .method = method_for(LifecycleStage::kStart)}};
  input.direct_edges = {{.declaration_scope_id = 7U,
                         .before_node_id = 1U,
                         .before_stage = LifecycleStage::kConfigure,
                         .after_node_id = 2U,
                         .after_stage = LifecycleStage::kConfigure}};

  // Key setup: shared helper asserts absence of a value and checks the error code.
  const auto expect_rejection = [](const LifecycleStageGraphInput& candidate,
                                   LifecycleKernelErrorCode expected) {
    const auto result = detail::build_lifecycle_stage_graph(candidate, capacity_for(candidate));
    if (result.has_value()) {
      ADD_FAILURE() << "malformed lifecycle input was accepted";
      return;
    }
    EXPECT_EQ(result.error().code, expected);
  };

  auto zero_scope = input;
  zero_scope.scopes[0].scope_id = 0U;
  expect_rejection(zero_scope, LifecycleKernelErrorCode::kInvalidScope);

  auto duplicate_scope = input;
  duplicate_scope.scopes.push_back({.scope_id = 7U});
  expect_rejection(duplicate_scope, LifecycleKernelErrorCode::kInvalidScope);

  auto missing_parent = input;
  missing_parent.scopes[0].parent_scope_id = 99U;
  expect_rejection(missing_parent, LifecycleKernelErrorCode::kInvalidScope);

  auto cyclic_scopes = input;
  cyclic_scopes.scopes = {
      {.scope_id = 7U, .parent_scope_id = 8U},
      {.scope_id = 8U, .parent_scope_id = 7U},
  };
  expect_rejection(cyclic_scopes, LifecycleKernelErrorCode::kInvalidScope);

  auto zero_node = input;
  zero_node.nodes[0].node_id = 0U;
  expect_rejection(zero_node, LifecycleKernelErrorCode::kInvalidNode);

  auto duplicate_node = input;
  duplicate_node.nodes[1].node_id = duplicate_node.nodes[0].node_id;
  expect_rejection(duplicate_node, LifecycleKernelErrorCode::kDuplicateNode);

  auto ambiguous_node = input;
  ambiguous_node.nodes[1].stable_name = ambiguous_node.nodes[0].stable_name;
  expect_rejection(ambiguous_node, LifecycleKernelErrorCode::kAmbiguousNodeIdentity);

  auto duplicate_callback = input;
  duplicate_callback.callbacks.push_back(duplicate_callback.callbacks.front());
  expect_rejection(duplicate_callback, LifecycleKernelErrorCode::kDuplicateCallback);

  auto invalid_callback_stage = input;
  invalid_callback_stage.callbacks[0].stage = std::bit_cast<LifecycleStage>(std::uint8_t{255U});
  expect_rejection(invalid_callback_stage, LifecycleKernelErrorCode::kInvalidStage);

  auto invalid_callback_method = input;
  invalid_callback_method.callbacks[0].method = method_for(LifecycleStage::kConfigure);
  expect_rejection(invalid_callback_method, LifecycleKernelErrorCode::kInvalidCallback);

  auto invalid_callback_origin = input;
  invalid_callback_origin.callbacks[0].origin =
      std::bit_cast<detail::LifecycleLogicalCallbackOrigin>(std::uint8_t{255U});
  expect_rejection(invalid_callback_origin, LifecycleKernelErrorCode::kInvalidCallback);

  auto invalid_legacy_stage = input;
  invalid_legacy_stage.callbacks[0] = {
      .node_id = 1U,
      .stage = LifecycleStage::kConfigure,
      .required = true,
      .method = method_for(LifecycleStage::kConfigure),
      .origin = detail::LifecycleLogicalCallbackOrigin::kLegacyStart,
  };
  expect_rejection(invalid_legacy_stage, LifecycleKernelErrorCode::kInvalidCallback);

  auto optional_legacy_stop = input;
  optional_legacy_stop.callbacks[0] = {
      .node_id = 1U,
      .stage = LifecycleStage::kStop,
      .required = false,
      .method = method_for(LifecycleStage::kStop),
      .origin = detail::LifecycleLogicalCallbackOrigin::kLegacyStop,
  };
  expect_rejection(optional_legacy_stop, LifecycleKernelErrorCode::kInvalidCallback);

  auto cross_stage_edge = input;
  cross_stage_edge.direct_edges[0].after_stage = LifecycleStage::kStart;
  expect_rejection(cross_stage_edge, LifecycleKernelErrorCode::kCrossStageDependency);
}

// REQ: LIFECYCLE-007
// Graph under test:
//
//   Scope 7
//   [first:1] --[configured fact]--> [second:2]  (kConfigure)
//   [first:1] --[direct edge]------> [second:2]  (kStart)
//
// Each capacity dimension is independently exhausted; all must yield kCapacityExceeded.
TEST(LifecycleDependencyPlan, RejectsEveryFiniteNormalizerCapacityDimension) {
  // Key setup: fully valid two-node input with both a fact dependency and a direct edge
  // so that all six capacity dimensions are exercised.
  LifecycleStageGraphInput input;
  input.scopes = {{.scope_id = 7U}};
  input.nodes = {
      {.node_id = 1U, .scope_id = 7U, .stable_name = "first", .stable_order = 1U},
      {.node_id = 2U, .scope_id = 7U, .stable_name = "second", .stable_order = 2U},
  };
  input.callbacks = {{.node_id = 1U,
                      .stage = LifecycleStage::kStart,
                      .method = method_for(LifecycleStage::kStart)}};
  input.producers = {
      {.node_id = 1U,
       .declared_stage = LifecycleStage::kConfigure,
       .key = {.scope_id = 7U, .stage = LifecycleStage::kConfigure, .stable_name = "ready"}}};
  input.requirements = {
      {.node_id = 2U,
       .declared_stage = LifecycleStage::kConfigure,
       .key = {.scope_id = 7U, .stage = LifecycleStage::kConfigure, .stable_name = "ready"}}};
  input.direct_edges = {{.declaration_scope_id = 7U,
                         .before_node_id = 1U,
                         .before_stage = LifecycleStage::kStart,
                         .after_node_id = 2U,
                         .after_stage = LifecycleStage::kStart}};

  struct CapacityCase {
    const char* name;
    void (*exhaust)(LifecycleKernelCapacity&);
  };
  constexpr std::array cases{
      CapacityCase{"scopes", [](LifecycleKernelCapacity& value) { value.max_scopes = 0U; }},
      CapacityCase{"nodes", [](LifecycleKernelCapacity& value) { value.max_nodes = 1U; }},
      CapacityCase{"callbacks", [](LifecycleKernelCapacity& value) { value.max_callbacks = 0U; }},
      CapacityCase{"dependency records",
                   [](LifecycleKernelCapacity& value) { value.max_dependency_records = 1U; }},
      CapacityCase{"direct edges",
                   [](LifecycleKernelCapacity& value) { value.max_direct_edges = 0U; }},
      CapacityCase{"completions",
                   [](LifecycleKernelCapacity& value) { value.max_completions = 11U; }},
  };

  // Key assertion: every under-sized capacity dimension independently yields kCapacityExceeded.
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    auto exhausted = capacity_for(input);
    test_case.exhaust(exhausted);
    const auto rejected = detail::build_lifecycle_stage_graph(input, exhausted);
    ASSERT_FALSE(rejected.has_value());
    EXPECT_EQ(rejected.error().code, LifecycleKernelErrorCode::kCapacityExceeded);
  }
}

}  // namespace
}  // namespace holoscan::test
