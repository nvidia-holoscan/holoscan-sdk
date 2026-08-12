/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include <unistd.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "compile_test_graph.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/execution_plan_state.hpp"
#include "compiler/process_placement_internal.hpp"
#include "compiler/runtime_package_capabilities.hpp"
#include "core/graph_compiler_access.hpp"
#include "interprocess_test_operators.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_worker.hpp"

namespace holoscan::test {
namespace {

// Runtime-materializing tests publish into the host-wide HoloMQ namespace. Give each invocation
// its own graph identity so concurrent CTest runs and parallel SDK containers cannot derive the
// same service names. Static compiler tests below keep their fixed graph names to verify
// deterministic lowering.
[[nodiscard]] std::string unique_runtime_graph_name(const std::string_view base) {
  static std::atomic<std::uint64_t> counter{};
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::string(base) + "-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
         std::to_string(timestamp) + "-" +
         std::to_string(counter.fetch_add(1U, std::memory_order_relaxed));
}

// Places "source" in "producer-process" and "target" in "consumer-process".
// Used by every linear cross-process test in this file.
[[nodiscard]] detail::ProcessPlacementBinding split_source_target_placement() {
  constexpr std::array first{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array second{detail::GraphObjectSelector{.graph_path = "target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = first,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumer",
                         detail::ProcessPlacement{
                             .members = second,
                             .launch_group = "consumer-process",
                         });
  return placement;
}

// Places "source" in "producer-process" and both "first_target"/"second_target"
// together in "consumer-process". Used by AcceptedCoverPreservesLocalityAndLogicalEdgeIdentity.
[[nodiscard]] detail::ProcessPlacementBinding split_fanout_placement() {
  constexpr std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array consumer_selectors{
      detail::GraphObjectSelector{.graph_path = "first_target"},
      detail::GraphObjectSelector{.graph_path = "second_target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("consumers",
                         detail::ProcessPlacement{
                             .members = consumer_selectors,
                             .launch_group = "consumer-process",
                         });
  return placement;
}

// Maps HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM to the canonical algorithm key
// expected in the compiled ExecutionPlan. Defaults to serial_baseline when unset.
[[nodiscard]] std::string_view selected_partition_algorithm() {
  const char* const selector = std::getenv("HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM");
  if (selector == nullptr || std::string_view(selector) == "serial_baseline/v1") {
    return "holoscan.partition.serial_baseline";
  }
  if (std::string_view(selector) == "source_chain/v1") {
    return "holoscan.partition.source_chain";
  }
  return {};
}

// One source fanning out to two independent consumers; used in
// CompatibleFanoutSharesOneDeterministicService and
// AcceptedCoverPreservesLocalityAndLogicalEdgeIdentity.
//
//   source --> first_target
//          --> second_target
class FanoutSourceGraph final : public Graph {
 public:
  explicit FanoutSourceGraph() : Graph("cross-process-fanout") {}

 protected:
  void compose() override {
    add_compile_test_default_clock(*this);
    auto source = op<CompileTestOperator>("source");
    auto first = op<CompileTestOperator>("first_target");
    auto second = op<CompileTestOperator>("second_target");
    add_flow(source->output, first->input);
    add_flow(source->output, second->input);
  }
};

class ConfiguredFanoutSourceGraph final : public Graph {
 public:
  explicit ConfiguredFanoutSourceGraph(const QueuePolicy second_policy)
      : Graph("configured-cross-process-fanout"), second_policy_(second_policy) {}

 protected:
  void compose() override {
    add_compile_test_default_clock(*this);
    auto source = op<CompileTestOperator>("source");
    auto first = op<CompileTestOperator>("first_target");
    auto second = op<CompileTestOperator>("second_target");
    add_flow(source->output,
             first->input,
             ConnectionOptions{.queue_policy = QueuePolicy::kBlocking, .queue_depth = 1U});
    add_flow(source->output,
             second->input,
             ConnectionOptions{.queue_policy = second_policy_, .queue_depth = 4U});
  }

 private:
  QueuePolicy second_policy_;
};

// Thin accessor that unwraps the internal ExecutionPlanState from a compiled plan.
[[nodiscard]] const detail::ExecutionPlanState* plan_state(const ExecutionPlan& plan) {
  return detail::ExecutionPlanAccess::state(plan);
}

}  // namespace

// One cross-process edge produces exactly one deterministic service/publisher/subscriber
// triple and records the HoloMQ transport capability requirement. Replay compilation and
// Profile::kResearch must not change the transport evidence identity.
//
//   source --[cross-process]--> target
//
// REQ: CHAN-085
TEST(InterprocessPartitionBoundaryTest, CrossProcessEdgeLowersOneDeterministicService) {
  Graph graph{"cross-process-linear"};
  add_linear_graph(graph);

  // Key setup: compile with split options so the single edge crosses process groups.
  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, split_source_target_placement());
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->payload_edges.size(), 1U);
  ASSERT_EQ(state->payload_edges.front().locality, ChannelLocalityKind::kCrossProcess);

  // Key assertion: one deterministic service with strictly separated process groups.
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  EXPECT_EQ(service.edge, state->payload_edges.front().edge);
  EXPECT_EQ(service.source, state->payload_edges.front().source);
  EXPECT_EQ(service.target, state->payload_edges.front().target);
  EXPECT_FALSE(service.service_id.empty());
  EXPECT_NE(service.service_id.find("cross-process-channel."), std::string::npos);
  EXPECT_EQ(service.publisher_history_depth, 0U);
  EXPECT_TRUE(service.event_wake);
  EXPECT_EQ(service.max_subscribers, 1U);
  EXPECT_EQ(service.max_subscriber_queue_depth, state->payload_edges.front().queue_depth);
  EXPECT_GE(service.max_subscriber_borrowed_samples, 1U);
  EXPECT_TRUE(service.source_process_group);
  EXPECT_TRUE(service.target_process_group);
  EXPECT_NE(service.source_process_group, service.target_process_group);

  ASSERT_EQ(state->cross_process_channel_publishers.size(), 1U);
  const auto& publisher = state->cross_process_channel_publishers.front();
  EXPECT_EQ(publisher.edge, service.edge);
  EXPECT_EQ(publisher.service_id, service.service_id);
  EXPECT_GE(publisher.max_loaned_samples, 1U);

  ASSERT_EQ(state->cross_process_channel_subscribers.size(), 1U);
  const auto& subscriber = state->cross_process_channel_subscribers.front();
  EXPECT_EQ(subscriber.edge, service.edge);
  EXPECT_EQ(subscriber.service_id, service.service_id);
  EXPECT_GE(subscriber.requested_queue_depth, 1U);
  EXPECT_GE(subscriber.readiness_capacity, subscriber.requested_queue_depth);
  EXPECT_TRUE(subscriber.event_wake);

  // The process-local family owns no cross-process edge and vice versa.
  EXPECT_TRUE(state->process_local_channel_services.empty());

  const auto transport_requirement = std::find_if(
      plan.profile_manifest().required_capabilities.begin(),
      plan.profile_manifest().required_capabilities.end(),
      [](const auto& requirement) {
        return requirement.capability_id == "holoscan.runtime.channel.holomq_transport";
      });
  ASSERT_NE(transport_requirement, plan.profile_manifest().required_capabilities.end());
  EXPECT_EQ(transport_requirement->version, 1U);
  EXPECT_EQ(transport_requirement->evidence_id, "holoscan.channel.holomq_transport/v1");

  // Deterministic replay: an identical authoring produces identical contracts.
  Graph replay_graph{"cross-process-linear"};
  add_linear_graph(replay_graph);
  const ExecutionPlan replay = detail::compile_with_process_placement(
      replay_graph, CompileOptions{}, split_source_target_placement());
  ASSERT_TRUE(replay.ok()) << replay.json();
  const auto* replay_state = plan_state(replay);
  ASSERT_NE(replay_state, nullptr);
  ASSERT_EQ(replay_state->cross_process_channel_services.size(), 1U);
  const auto& replayed = replay_state->cross_process_channel_services.front();
  EXPECT_EQ(replay.digest(), plan.digest());
  EXPECT_EQ(replayed.service_id, service.service_id);
  EXPECT_EQ(replayed.queue_policy, service.queue_policy);
  EXPECT_EQ(replayed.max_subscriber_queue_depth, service.max_subscriber_queue_depth);
  EXPECT_EQ(replayed.payload_size_bytes, service.payload_size_bytes);
  EXPECT_EQ(plan.json().find(service.service_id), std::string::npos);

  // Profile policy may accept or reject a graph, but it never selects a
  // different transport implementation. Every admitted cross-process plan
  // asks the same fixed package provider for the same exact transport offer.
  Graph research_graph{"cross-process-research-profile"};
  add_linear_graph(research_graph);
  CompileOptions research_options;
  research_options.profile = Profile::kResearch;
  const ExecutionPlan research_plan = detail::compile_with_process_placement(
      research_graph, std::move(research_options), split_source_target_placement());
  ASSERT_TRUE(research_plan.ok()) << research_plan.json();
  const auto research_transport = std::find_if(
      research_plan.profile_manifest().required_capabilities.begin(),
      research_plan.profile_manifest().required_capabilities.end(),
      [](const auto& requirement) {
        return requirement.capability_id == "holoscan.runtime.channel.holomq_transport";
      });
  ASSERT_NE(research_transport, research_plan.profile_manifest().required_capabilities.end());
  EXPECT_EQ(research_transport->version, transport_requirement->version);
  EXPECT_EQ(research_transport->evidence_id, transport_requirement->evidence_id);
}

// Each fan-out leg lands in its own dedicated process group. Compatible arms from
// the same source and policy share one broadcast service while retaining one
// subscriber record per logical edge.
//
//   source --[cross-process]--> first_target  (first-consumer-process)
//          --[cross-process]--> second_target (second-consumer-process)
//
// REQ: CHAN-085
// REQ: CHAN-112
TEST(InterprocessPartitionBoundaryTest, CompatibleFanoutSharesOneDeterministicService) {
  FanoutSourceGraph graph;
  constexpr std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array first_selector{detail::GraphObjectSelector{.graph_path = "first_target"}};
  constexpr std::array second_selector{detail::GraphObjectSelector{.graph_path = "second_target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("first-consumer",
                         detail::ProcessPlacement{
                             .members = first_selector,
                             .launch_group = "first-consumer-process",
                         })
      .bind_process_role("second-consumer",
                         detail::ProcessPlacement{
                             .members = second_selector,
                             .launch_group = "second-consumer-process",
                         });

  // Key setup: three separate process groups exercise one service across two consumer workers.
  const ExecutionPlan plan =
      detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
  ASSERT_TRUE(plan.ok()) << plan.json();

  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  // Key assertion: one publisher service covers both edge-local subscribers and target groups.
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  EXPECT_EQ(service.source, state->payload_edges.front().source);
  EXPECT_EQ(service.max_subscribers, 2U);
  EXPECT_EQ(service.target_process_groups.size(), 2U);
  EXPECT_EQ(state->cross_process_channel_publishers.size(), 1U);
  EXPECT_EQ(state->cross_process_channel_subscribers.size(), 2U);
  EXPECT_EQ(state->cross_process_channel_publishers.front().service_id, service.service_id);
  EXPECT_TRUE(std::ranges::all_of(state->cross_process_channel_subscribers, [&](const auto& item) {
    return item.service_id == service.service_id;
  }));
}

// Compatible arms may request different queue depths because the service records
// only the admitted ceiling and each subscriber retains its exact allocation.
// Changing a service-static queue policy instead creates a separate service.
//
//   source --[blocking, depth=1]--> first_target
//          --[policy,   depth=4]--> second_target
//
// REQ: CHAN-085
TEST(InterprocessPartitionBoundaryTest, FanoutCoalescingPreservesQueueGeometryAndPolicySplits) {
  // Key setup: matching policies use one service despite unequal per-arm depths.
  ConfiguredFanoutSourceGraph compatible{QueuePolicy::kBlocking};
  const ExecutionPlan compatible_plan = detail::compile_with_process_placement(
      compatible, CompileOptions{}, split_fanout_placement());
  ASSERT_TRUE(compatible_plan.ok()) << compatible_plan.json();
  const auto* compatible_state = plan_state(compatible_plan);
  ASSERT_NE(compatible_state, nullptr);
  ASSERT_EQ(compatible_state->cross_process_channel_services.size(), 1U);
  ASSERT_EQ(compatible_state->cross_process_channel_subscribers.size(), 2U);

  const auto& shared_service = compatible_state->cross_process_channel_services.front();
  EXPECT_EQ(shared_service.max_subscriber_queue_depth, 4U);
  EXPECT_EQ(shared_service.max_subscriber_borrowed_samples, 4U);
  std::array<std::uint32_t, 2U> requested_depths{
      compatible_state->cross_process_channel_subscribers[0].requested_queue_depth,
      compatible_state->cross_process_channel_subscribers[1].requested_queue_depth,
  };
  std::ranges::sort(requested_depths);
  EXPECT_EQ(requested_depths, (std::array<std::uint32_t, 2U>{1U, 4U}));

  // Key assertion: a differing queue policy cannot reuse the blocking service contract.
  ConfiguredFanoutSourceGraph incompatible{QueuePolicy::kDropOldest};
  const ExecutionPlan incompatible_plan = detail::compile_with_process_placement(
      incompatible, CompileOptions{}, split_fanout_placement());
  ASSERT_TRUE(incompatible_plan.ok()) << incompatible_plan.json();
  const auto* incompatible_state = plan_state(incompatible_plan);
  ASSERT_NE(incompatible_state, nullptr);
  ASSERT_EQ(incompatible_state->cross_process_channel_services.size(), 2U);
  ASSERT_EQ(incompatible_state->cross_process_channel_publishers.size(), 2U);
  ASSERT_EQ(incompatible_state->cross_process_channel_subscribers.size(), 2U);
  EXPECT_NE(incompatible_state->cross_process_channel_services[0].queue_policy,
            incompatible_state->cross_process_channel_services[1].queue_policy);
  EXPECT_NE(incompatible_state->cross_process_channel_services[0].service_id,
            incompatible_state->cross_process_channel_services[1].service_id);
}

// Both consumers share one process group (split_fanout_placement). The accepted
// partition cover keeps two distinct cross-process edges and subscriber records,
// but compatible arms share one service and publisher.
//
//   source --[cross-process]--> first_target  \
//          --[cross-process]--> second_target  } both in consumer-process
//
// Planning references: SP32, SP36, SP47, SP-T15, SP-T41.
// REQ: CHAN-085
TEST(InterprocessPartitionBoundaryTest, AcceptedCoverPreservesLocalityAndLogicalEdgeIdentity) {
  // Key setup: split_fanout_placement collapses both consumers into one process group
  // while keeping the source separate, producing two cross-process edges.
  FanoutSourceGraph graph;
  const ExecutionPlan plan =
      detail::compile_with_process_placement(graph, CompileOptions{}, split_fanout_placement());
  ASSERT_TRUE(plan.ok()) << plan.json();

  const std::string_view expected_algorithm = selected_partition_algorithm();
  ASSERT_FALSE(expected_algorithm.empty());
  ASSERT_TRUE(plan.partitioning_plan().accepted_algorithm.has_value());
  const AlgorithmKey accepted_algorithm =
      plan.partitioning_plan().accepted_algorithm.value_or(AlgorithmKey{});
  EXPECT_EQ(accepted_algorithm.id, expected_algorithm);
  EXPECT_EQ(accepted_algorithm.version, 1U);
  EXPECT_EQ(plan.scheduler_plan().ready_policy.id, "holoscan.ready.deterministic_serial");
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, plan.execution_partitions().size());
  EXPECT_EQ(plan.scheduler_plan().activation_overlap, plan.execution_partitions().size() > 1U);

  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->payload_edges.size(), 2U);
  ASSERT_EQ(state->channel_locality.size(), 2U);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  ASSERT_EQ(state->cross_process_channel_publishers.size(), 1U);
  ASSERT_EQ(state->cross_process_channel_subscribers.size(), 2U);
  ASSERT_EQ(plan.schedule_edges().size(), 2U);

  std::set<std::uint64_t> payload_edges;
  std::set<std::uint64_t> locality_edges;
  std::set<std::uint64_t> subscriber_edges;
  std::set<std::uint64_t> schedule_edges;
  std::set<std::string> service_ids;
  for (const PayloadEdgePlanRecord& edge : state->payload_edges) {
    EXPECT_EQ(edge.locality, ChannelLocalityKind::kCrossProcess);
    payload_edges.insert(edge.edge.value);
  }
  for (const ChannelLocalityRecord& locality : state->channel_locality) {
    EXPECT_EQ(locality.requested, ChannelLocalityKind::kCrossProcess);
    EXPECT_EQ(locality.accepted, ChannelLocalityKind::kCrossProcess);
    locality_edges.insert(locality.edge.value);
  }
  for (const auto& subscriber : state->cross_process_channel_subscribers) {
    subscriber_edges.insert(subscriber.edge.value);
    service_ids.insert(subscriber.service_id);
  }
  for (const ScheduleEdgeRecord& edge : plan.schedule_edges()) {
    EXPECT_NE(edge.source, edge.destination);
    schedule_edges.insert(edge.edge.value);
  }

  // Key assertion: edge-local inventories remain exact while both arms name one shared service.
  EXPECT_EQ(payload_edges, locality_edges);
  EXPECT_EQ(payload_edges, subscriber_edges);
  EXPECT_EQ(payload_edges, schedule_edges);
  EXPECT_EQ(service_ids.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  EXPECT_EQ(service.max_subscribers, 2U);
  ASSERT_EQ(service.target_process_groups.size(), 1U);
  EXPECT_EQ(service.target_process_groups.front(), service.target_process_group);

  // The baseline uses one partition per process domain. Source-chain is free
  // to split the two fan-out legs, but both covers retain the same two
  // cross-process logical edges and their deterministic shared service.
  if (expected_algorithm == "holoscan.partition.serial_baseline") {
    EXPECT_EQ(plan.execution_partitions().size(), 2U);
  } else {
    EXPECT_EQ(plan.execution_partitions().size(), 3U);
  }
}
namespace {

using namespace std::chrono_literals;

// Shared static counters and a condition variable used to coordinate producer and
// consumer workers in CompiledCrossProcessEdgeCarriesDataThroughItsService.
struct CrossProcessFloorProbe {
  static void reset() noexcept {
    count.store(0U, std::memory_order_release);
    last.store(0U, std::memory_order_release);
    source_starts.store(0U, std::memory_order_release);
    sink_starts.store(0U, std::memory_order_release);
  }

  static void record(const std::uint32_t value) noexcept {
    {
      const std::lock_guard lock{mutex};
      last.store(value, std::memory_order_release);
      count.fetch_add(1U, std::memory_order_acq_rel);
    }
    ready.notify_all();
  }

  static void record_source_start() noexcept {
    source_starts.fetch_add(1U, std::memory_order_acq_rel);
    ready.notify_all();
  }

  static void record_sink_start() noexcept {
    sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 10s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  [[nodiscard]] static bool wait_for_starts(const std::size_t expected_source,
                                            const std::size_t expected_sink) {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [=] {
      return source_starts.load(std::memory_order_acquire) >= expected_source &&
             sink_starts.load(std::memory_order_acquire) >= expected_sink;
    });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::atomic<std::uint32_t> last{};
  static inline std::atomic<std::size_t> source_starts{};
  static inline std::atomic<std::size_t> sink_starts{};
};

[[nodiscard]] ScalarProcessBoundaryHooks cross_process_floor_hooks() noexcept {
  return {
      .payload = 0xfeedbeefU,
      .source_started = &CrossProcessFloorProbe::record_source_start,
      .sink_started = &CrossProcessFloorProbe::record_sink_start,
      .sample_received = &CrossProcessFloorProbe::record,
  };
}

struct CrossProcessFanoutProbe {
  static constexpr std::size_t kSampleCount = 8U;

  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    source_starts = 0U;
    sink_starts = 0U;
    emitted = 0U;
    counts = {};
    values = {};
  }

  static void record_source_start() {
    {
      const std::lock_guard lock{mutex};
      ++source_starts;
    }
    ready.notify_all();
  }

  static void record_sink_start(std::size_t) {
    {
      const std::lock_guard lock{mutex};
      ++sink_starts;
    }
    ready.notify_all();
  }

  static void record_emit() {
    {
      const std::lock_guard lock{mutex};
      ++emitted;
    }
    ready.notify_all();
  }

  static void record(const std::size_t sink, const std::uint32_t value) {
    {
      const std::lock_guard lock{mutex};
      if (sink >= counts.size()) {
        return;
      }
      if (counts[sink] < kSampleCount) {
        values[sink][counts[sink]] = value;
      }
      ++counts[sink];
    }
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_sink_starts() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return sink_starts == 2U; });
  }

  [[nodiscard]] static bool wait_for_delivery() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] {
      return emitted == kSampleCount && counts[0] == kSampleCount && counts[1] == kSampleCount;
    });
  }

  [[nodiscard]] static bool exact_sequences() {
    const std::lock_guard lock{mutex};
    const auto expected = [] {
      std::array<std::uint32_t, kSampleCount> sequence{};
      for (std::size_t index = 0U; index < sequence.size(); ++index) {
        sequence[index] = static_cast<std::uint32_t>(index + 1U);
      }
      return sequence;
    }();
    return source_starts == 1U && sink_starts == 2U && emitted == kSampleCount &&
           counts[0] == kSampleCount && counts[1] == kSampleCount && values[0] == expected &&
           values[1] == expected;
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::size_t source_starts{};
  static inline std::size_t sink_starts{};
  static inline std::size_t emitted{};
  static inline std::array<std::size_t, 2U> counts{};
  static inline std::array<std::array<std::uint32_t, kSampleCount>, 2U> values{};
};

[[nodiscard]] FanoutProcessBoundaryHooks cross_process_fanout_hooks() noexcept {
  return {
      .sample_count = CrossProcessFanoutProbe::kSampleCount,
      .source_started = &CrossProcessFanoutProbe::record_source_start,
      .sink_started = &CrossProcessFanoutProbe::record_sink_start,
      .sample_emitted = &CrossProcessFanoutProbe::record_emit,
      .sample_received = &CrossProcessFanoutProbe::record,
  };
}

// Shared static state capturing the last received Tensor's byte value, read-only flag,
// and device type. Coordinates producer/consumer startup and sample delivery for the
// host and pinned-host Tensor cross-process tests.
struct CrossProcessTensorProbe {
  static void reset() noexcept {
    count.store(0U, std::memory_order_release);
    value.store(0U, std::memory_order_release);
    read_only.store(false, std::memory_order_release);
    ordered.store(true, std::memory_order_release);
    device_type.store(static_cast<std::int32_t>(kDLExtDev), std::memory_order_release);
    source_starts.store(0U, std::memory_order_release);
    sink_starts.store(0U, std::memory_order_release);
  }

  static void record(const Tensor& tensor) noexcept {
    const auto* observed = tensor.data_as<std::uint64_t>();
    const std::uint64_t observed_value = observed != nullptr ? *observed : 0U;
    const std::size_t index = count.load(std::memory_order_relaxed);
    value.store(observed_value, std::memory_order_release);
    if (observed_value != kBaseValue + index) {
      ordered.store(false, std::memory_order_release);
    }
    read_only.store(tensor.is_read_only(), std::memory_order_release);
    device_type.store(static_cast<std::int32_t>(tensor.device().device_type),
                      std::memory_order_release);
    count.fetch_add(1U, std::memory_order_acq_rel);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_starts() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] {
      return source_starts.load(std::memory_order_acquire) == 1U &&
             sink_starts.load(std::memory_order_acquire) == 1U;
    });
  }

  [[nodiscard]] static bool wait_for_sink_start() {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 10s, [] { return sink_starts.load(std::memory_order_acquire) == 1U; });
  }

  [[nodiscard]] static bool wait_for_sample() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return count.load(std::memory_order_acquire) != 0U; });
  }

  [[nodiscard]] static bool wait_for_sample_count(std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 10s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::atomic<std::uint64_t> value{};
  static inline std::atomic_bool read_only{};
  static inline std::atomic_bool ordered{true};
  static inline std::atomic<std::int32_t> device_type{};
  static inline std::atomic<std::size_t> source_starts{};
  static inline std::atomic<std::size_t> sink_starts{};
  static constexpr std::uint64_t kBaseValue = 0x5a5a5a5a5a5a5a5aULL;
};

// Template source that allocates one Tensor loan directly into the HoloMQ pool and
// emits only the resulting descriptor. Storage=kHost uses a host write; Storage=kPinnedHost
// uses a cudaMemset+cudaMemcpy on the compute stream before committing the loan.
// Instantiated as CrossProcessTensorSource and CrossProcessPinnedTensorSource.
template <MemoryKind Storage, std::size_t SampleCount = 1U>
class CrossProcessHostTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorPortLayout{
            .memory_kind = Storage,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 0U,
        })
        .tensor_allocation(
            TensorAllocationBounds{.max_rank = 1U, .max_byte_span = sizeof(std::uint64_t)});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    if constexpr (SampleCount == 1U) {
      result.trigger(OnClock{.period = 20ms});
    } else {
      result.trigger(OnClock{.period = 1ms});
    }
    return result;
  }

  void start() override {
    if constexpr (Storage == MemoryKind::kPinnedHost) {
      if (cudaSetDevice(0) != cudaSuccess ||
          cudaMalloc(&device_value_, sizeof(std::uint64_t)) != cudaSuccess) {
        if (device_value_ != nullptr) {
          static_cast<void>(cudaFree(device_value_));
          device_value_ = nullptr;
        }
      }
    }
    CrossProcessTensorProbe::source_starts.fetch_add(1U, std::memory_order_acq_rel);
    CrossProcessTensorProbe::ready.notify_all();
  }

  void stop() override {
    if (device_value_ != nullptr) {
      static_cast<void>(cudaFree(device_value_));
      device_value_ = nullptr;
    }
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (published_count_ == SampleCount) {
      return {};
    }
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    });
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan).error());
    }
    auto writer = [&]() {
      if constexpr (Storage == MemoryKind::kPinnedHost) {
        return loan->write(context.cuda_stream());
      }
      return loan->write_host();
    }();
    if (!writer) {
      return make_unexpected(std::move(writer).error());
    }
    if constexpr (Storage == MemoryKind::kPinnedHost) {
      // Produce the device value and copy it into the final pinned transport slot on one stream.
      if (device_value_ == nullptr || context.cuda_stream().get() == nullptr ||
          cudaMemsetAsync(
              device_value_, kFillByte, sizeof(std::uint64_t), context.cuda_stream().get()) !=
              cudaSuccess ||
          cudaMemcpyAsync(writer->data(),
                          device_value_,
                          sizeof(std::uint64_t),
                          cudaMemcpyDeviceToHost,
                          context.cuda_stream().get()) != cudaSuccess) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    } else {
      if (writer->template data_as<std::uint64_t>() == nullptr) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      *writer->template data_as<std::uint64_t>() = kValue + published_count_;
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted && (emitted.error().code == ErrorCode::kBackpressured ||
                     emitted.error().code == ErrorCode::kNotReady ||
                     emitted.error().code == ErrorCode::kPublicationIndeterminate)) {
      return {};
    }
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    ++published_count_;
    return {};
  }

  Output<Tensor> output;
  static constexpr std::uint8_t kFillByte = 0x5aU;
  static constexpr std::uint64_t kValue = CrossProcessTensorProbe::kBaseValue;

 private:
  std::size_t published_count_{};
  void* device_value_{};
};

// Receives the imported read-only Tensor loan, synchronizes for host access when
// Storage=kPinnedHost, and records it via CrossProcessTensorProbe.
// Instantiated as CrossProcessTensorSink and CrossProcessPinnedTensorSink.
template <MemoryKind Storage>
class CrossProcessHostTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorPortLayout{
            .memory_kind = Storage,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 0U,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  void start() override {
    CrossProcessTensorProbe::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    CrossProcessTensorProbe::ready.notify_all();
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    if constexpr (Storage == MemoryKind::kPinnedHost) {
      auto ready = sample->data.synchronize_for_host();
      if (!ready) {
        return make_unexpected(std::move(ready).error());
      }
    }
    CrossProcessTensorProbe::record(sample->data);
    return {};
  }

  Input<Tensor> input;
};

using CrossProcessTensorSource = CrossProcessHostTensorSource<MemoryKind::kHost>;
using CrossProcessTensorSink = CrossProcessHostTensorSink<MemoryKind::kHost>;
using CrossProcessTensorBurstSource = CrossProcessHostTensorSource<MemoryKind::kHost, 8U>;
using CrossProcessPinnedTensorSource = CrossProcessHostTensorSource<MemoryKind::kPinnedHost>;
using CrossProcessPinnedTensorSink = CrossProcessHostTensorSink<MemoryKind::kPinnedHost>;

// Shared static state for the CUDA device Tensor cross-process test: records the
// byte value (copied host-side after synchronization) and the read-only flag.
struct CrossProcessDeviceTensorProbe {
  static void reset() noexcept {
    count.store(0U, std::memory_order_release);
    value.store(0U, std::memory_order_release);
    read_only.store(false, std::memory_order_release);
    source_starts.store(0U, std::memory_order_release);
    sink_starts.store(0U, std::memory_order_release);
  }

  static void record(std::uint64_t observed, bool observed_read_only) noexcept {
    value.store(observed, std::memory_order_release);
    read_only.store(observed_read_only, std::memory_order_release);
    count.fetch_add(1U, std::memory_order_acq_rel);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_sample() {
    std::unique_lock lock{mutex};
    return ready.wait_for(lock, 10s, [] { return count.load(std::memory_order_acquire) != 0U; });
  }

  [[nodiscard]] static bool wait_for_sink_start() {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 10s, [] { return sink_starts.load(std::memory_order_acquire) == 1U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::atomic<std::uint64_t> value{};
  static inline std::atomic_bool read_only{};
  static inline std::atomic<std::size_t> source_starts{};
  static inline std::atomic<std::size_t> sink_starts{};
};

// Emits a single CUDA device Tensor loan filled via cudaMemsetAsync on the compute
// stream. The generation fence is recorded by HoloMQ before publication; the sink
// must observe the fence via synchronize_for_host before reading back the bytes.
// Used in DirectCudaTensorLoanPreservesDeviceOrderingAcrossGroups.
class CrossProcessDeviceTensorSource final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.output(output, "output")
        .max_emits_per_compute(1U)
        .produces_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kCudaDevice,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 0U,
        })
        .tensor_allocation(
            TensorAllocationBounds{.max_rank = 1U, .max_byte_span = sizeof(std::uint64_t)});
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 20ms});
    return result;
  }

  void start() override {
    CrossProcessDeviceTensorProbe::source_starts.fetch_add(1U, std::memory_order_acq_rel);
    CrossProcessDeviceTensorProbe::ready.notify_all();
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    if (published_) {
      return {};
    }
    auto loan = output.allocate_tensor(TensorLoanRequest{
        .shape = TensorShapeView{},
        .dtype = DLDataType{kDLUInt, 64U, 1U},
    });
    if (!loan) {
      if (loan.error().code == ErrorCode::kResourceExhausted) {
        return {};
      }
      return make_unexpected(std::move(loan).error());
    }
    auto writer = loan->write(context.cuda_stream());
    if (!writer) {
      return make_unexpected(std::move(writer).error());
    }
    if (context.cuda_stream().get() == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    if (cudaMemsetAsync(
            writer->data(), kByteValue, writer->byte_size(), context.cuda_stream().get()) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto committed = std::move(*writer).commit();
    if (!committed) {
      return make_unexpected(std::move(committed).error());
    }
    auto emitted = output.emit(std::move(*loan));
    if (!emitted && (emitted.error().code == ErrorCode::kNotReady ||
                     emitted.error().code == ErrorCode::kBackpressured ||
                     emitted.error().code == ErrorCode::kPublicationIndeterminate)) {
      return {};
    }
    if (!emitted) {
      return make_unexpected(std::move(emitted).error());
    }
    published_ = true;
    return {};
  }

  Output<Tensor> output;
  static constexpr std::uint8_t kByteValue = 0x6bU;
  static constexpr std::uint64_t kValue = 0x6b6b6b6b6b6b6b6bULL;

 private:
  bool published_{};
};

// Receives the imported device Tensor read-only lease, calls synchronize_for_host
// to consume the generation fence, then copies the scalar value to the host for
// verification. Used in DirectCudaTensorLoanPreservesDeviceOrderingAcrossGroups.
class CrossProcessDeviceTensorSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(1U)
        .expects_tensor(TensorPortLayout{
            .memory_kind = MemoryKind::kCudaDevice,
            .dtype = DLDataType{kDLUInt, 64U, 1U},
            .rank = 0U,
        });
  }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }

  void start() override {
    CrossProcessDeviceTensorProbe::sink_starts.fetch_add(1U, std::memory_order_acq_rel);
    CrossProcessDeviceTensorProbe::ready.notify_all();
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    if (sample->data.data() == nullptr) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    auto ready = sample->data.synchronize_for_host();
    if (!ready) {
      return make_unexpected(std::move(ready).error());
    }
    std::uint64_t observed{};
    if (cudaMemcpy(&observed, sample->data.data(), sizeof(observed), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    CrossProcessDeviceTensorProbe::record(observed, sample->data.is_read_only());
    return {};
  }

  Input<Tensor> input;
};

}  // namespace

// End-to-end data-flow verification: a scalar uint32 must arrive in the consumer worker
// after the producer ticks. Also verifies that an unselected run_async (no worker filter)
// fails closed without starting either operator.
//
//   source --[cross-process, depth=4]--> target
//
// REQ: CHAN-085
TEST(InterprocessPartitionBoundaryTest, CompiledCrossProcessEdgeCarriesDataThroughItsService) {
  CrossProcessFloorProbe::reset();
  Graph graph{unique_runtime_graph_name("cross-process-floor")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<ScalarProcessBoundarySource>("source", cross_process_floor_hooks());
  auto target = graph.op<ScalarProcessBoundarySink>("target", cross_process_floor_hooks());
  graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 4U});

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, split_source_target_placement());
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);

  // Key setup: a multi-group plan must fail closed without a worker selector; only
  // targeted run_async_worker calls may start individual process group halves.
  // Application-author execution has no worker selector. A multi-group plan therefore fails
  // closed instead of reconstructing both graph halves in one process-local session.
  RunSession unselected = run_async(plan);
  unselected.wait();
  EXPECT_EQ(unselected.termination(), RunTermination::kFailed);
  EXPECT_EQ(CrossProcessFloorProbe::source_starts.load(std::memory_order_acquire), 0U);
  EXPECT_EQ(CrossProcessFloorProbe::sink_starts.load(std::memory_order_acquire), 0U);

  const auto& service = state->cross_process_channel_services.front();
  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  ASSERT_TRUE(CrossProcessFloorProbe::wait_for_starts(0U, 1U))
      << "termination=" << static_cast<int>(consumer.termination()) << " diagnostics="
      << (consumer.diagnostics().empty() ? "<none>" : consumer.diagnostics().front().token);
  EXPECT_EQ(CrossProcessFloorProbe::source_starts.load(std::memory_order_acquire), 0U);

  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  ASSERT_TRUE(CrossProcessFloorProbe::wait_for_starts(1U, 1U));
  // Key assertion: at least two samples arrive with the sentinel value and both workers
  // terminate cleanly with no diagnostics.
  const bool received = CrossProcessFloorProbe::wait_for_count(2U);
  producer.request_stop();
  consumer.request_stop();
  producer.wait();
  consumer.wait();
  EXPECT_TRUE(received) << "producer termination=" << static_cast<int>(producer.termination())
                        << " consumer termination=" << static_cast<int>(consumer.termination());
  EXPECT_EQ(CrossProcessFloorProbe::last.load(std::memory_order_acquire), 0xfeedbeefU);
  EXPECT_EQ(CrossProcessFloorProbe::source_starts.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(CrossProcessFloorProbe::sink_starts.load(std::memory_order_acquire), 1U);
  EXPECT_EQ(producer.termination(), RunTermination::kStopped);
  EXPECT_EQ(consumer.termination(), RunTermination::kStopped);
  EXPECT_TRUE(producer.diagnostics().empty())
      << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
  EXPECT_TRUE(consumer.diagnostics().empty())
      << (consumer.diagnostics().empty() ? "" : consumer.diagnostics().front().token);
}

// One compiler-frozen service broadcasts each logical publication to subscribers
// owned by two independently selected consumer worker sessions.
//
//   source --[shared cross-process service]--> first_target
//          --[shared cross-process service]--> second_target
//
// REQ: CHAN-085
TEST(InterprocessPartitionBoundaryTest, CoalescedFanoutDeliversExactSequenceToEveryWorker) {
  CrossProcessFanoutProbe::reset();
  Graph graph{unique_runtime_graph_name("cross-process-runtime-fanout")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<FanoutProcessBoundarySource>("source", cross_process_fanout_hooks());
  auto first =
      graph.op<FanoutProcessBoundarySink>("first_target", cross_process_fanout_hooks(), 0U);
  auto second =
      graph.op<FanoutProcessBoundarySink>("second_target", cross_process_fanout_hooks(), 1U);
  graph.add_flow(source->output, first->input, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->output, second->input, ConnectionOptions{.queue_depth = 4U});

  constexpr std::array source_selector{detail::GraphObjectSelector{.graph_path = "source"}};
  constexpr std::array first_selector{detail::GraphObjectSelector{.graph_path = "first_target"}};
  constexpr std::array second_selector{detail::GraphObjectSelector{.graph_path = "second_target"}};
  detail::ProcessPlacementBinding placement;
  placement
      .bind_process_role("producer",
                         detail::ProcessPlacement{
                             .members = source_selector,
                             .launch_group = "producer-process",
                         })
      .bind_process_role("first-consumer",
                         detail::ProcessPlacement{
                             .members = first_selector,
                             .launch_group = "first-consumer-process",
                         })
      .bind_process_role("second-consumer",
                         detail::ProcessPlacement{
                             .members = second_selector,
                             .launch_group = "second-consumer-process",
                         });

  // Key setup: attach both consumer workers before starting the sole publisher.
  const ExecutionPlan plan =
      detail::compile_with_process_placement(graph, CompileOptions{}, std::move(placement));
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  ASSERT_EQ(service.max_subscribers, 2U);
  ASSERT_EQ(service.target_process_groups.size(), 2U);

  RunSession first_consumer = detail::run_async_worker(plan, service.target_process_groups.front());
  RunSession second_consumer = detail::run_async_worker(plan, service.target_process_groups.back());
  ASSERT_TRUE(CrossProcessFanoutProbe::wait_for_sink_starts());
  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  const bool delivered = CrossProcessFanoutProbe::wait_for_delivery();

  producer.request_stop();
  first_consumer.request_stop();
  second_consumer.request_stop();
  producer.wait();
  first_consumer.wait();
  second_consumer.wait();

  // Key assertion: every accepted value appears exactly once and in order at both sinks.
  EXPECT_TRUE(delivered);
  EXPECT_TRUE(CrossProcessFanoutProbe::exact_sequences());
  EXPECT_EQ(producer.termination(), RunTermination::kStopped);
  EXPECT_EQ(first_consumer.termination(), RunTermination::kStopped);
  EXPECT_EQ(second_consumer.termination(), RunTermination::kStopped);
  EXPECT_TRUE(producer.diagnostics().empty())
      << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
  EXPECT_TRUE(first_consumer.diagnostics().empty())
      << (first_consumer.diagnostics().empty() ? "" : first_consumer.diagnostics().front().token);
  EXPECT_TRUE(second_consumer.diagnostics().empty())
      << (second_consumer.diagnostics().empty() ? "" : second_consumer.diagnostics().front().token);
}

// The matching process-local topology uses the same multi-recipient publication contract.
//
//   source --[shared process-local service]--> first_target
//          --[shared process-local service]--> second_target
//
// REQ: CHAN-085
// REQ: RUNTIME-029
TEST(InterprocessPartitionBoundaryTest, ProcessLocalFanoutDeliversExactSequenceToEverySink) {
  CrossProcessFanoutProbe::reset();
  Graph graph{unique_runtime_graph_name("process-local-runtime-fanout")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<FanoutProcessBoundarySource>("source", cross_process_fanout_hooks());
  auto first =
      graph.op<FanoutProcessBoundarySink>("first_target", cross_process_fanout_hooks(), 0U);
  auto second =
      graph.op<FanoutProcessBoundarySink>("second_target", cross_process_fanout_hooks(), 1U);
  graph.add_flow(source->output, first->input, ConnectionOptions{.queue_depth = 4U});
  graph.add_flow(source->output, second->input, ConnectionOptions{.queue_depth = 4U});

  // Key setup: compile one process group with a shared service and both subscriber arms.
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_TRUE(state->cross_process_channel_services.empty());
  ASSERT_EQ(state->process_local_channel_services.size(), 1U);
  EXPECT_EQ(state->process_local_channel_services.front().max_subscribers, 2U);

  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan observed_plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  ASSERT_TRUE(observed_plan.ok()) << observed_plan.json();
  RunSession session = run_async(observed_plan);
  const bool delivered = CrossProcessFanoutProbe::wait_for_delivery();
  session.request_stop();
  session.wait();

  // Key assertion: locality does not change exact ordered multi-recipient delivery.
  EXPECT_TRUE(delivered);
  EXPECT_TRUE(CrossProcessFanoutProbe::exact_sequences());
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);

  const auto snapshot = counters->snapshot();
  const auto count = [&](const detail::RuntimePhaseCounter counter) {
    return snapshot[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAttempts),
            CrossProcessFanoutProbe::kSampleCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitAccepted),
            CrossProcessFanoutProbe::kSampleCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelSamplesReceived),
            2U * CrossProcessFanoutProbe::kSampleCount);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelActivationCuts),
            count(detail::RuntimePhaseCounter::kChannelActivationCutDrains) +
                count(detail::RuntimePhaseCounter::kChannelActivationCutSkips));
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kChannelEmitBackpressured), 0U);
}

// A direct Tensor output in the producer worker loans the final HoloMQ host backing. Successful
// delivery proves publication exported a BufferHandle rather than staging or sending a local
// pointer, because the ordinary native direct-output reference cannot enter this transport arm.
//
//   source --[cross-process, host Tensor, depth=1]--> target
//
// REQ: CHAN-088
// REQ: CHAN-093
TEST(InterprocessPartitionBoundaryTest, DirectTensorLoanCrossesProcessGroupsWithoutPayloadCopy) {
  CrossProcessTensorProbe::reset();
  Graph graph{unique_runtime_graph_name("cross-process-direct-tensor")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<CrossProcessTensorSource>("source");
  auto target = graph.op<CrossProcessTensorSink>("target");
  graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 1U});

  // Key setup: compilation must bind the cross-process service to the direct output's
  // physical pool (service.tensor_pool_id == tensor_allocations.front().pool_id).
  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, split_source_target_placement());
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  ASSERT_EQ(state->tensor_allocations.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  EXPECT_EQ(service.tensor_pool_id, state->tensor_allocations.front().pool_id);
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kHost);
  // Key assertion: queue occupancy and a taken-but-retained imported sample have distinct
  // queue-depth-one contributors in addition to producer and retirement capacity.
  EXPECT_EQ(state->tensor_allocations.front().slot_count, 4U);
  const auto imported_retention =
      std::find_if(state->pool_capacity_contributors.begin(),
                   state->pool_capacity_contributors.end(),
                   [&](const auto& contributor) {
                     return contributor.pool_id == state->tensor_allocations.front().pool_id &&
                            contributor.contributor_id.ends_with(".transport-retention");
                   });
  ASSERT_NE(imported_retention, state->pool_capacity_contributors.end());
  EXPECT_EQ(imported_retention->maximum_distinct_slots, 1U);
  EXPECT_EQ(imported_retention->reasons,
            (std::vector{TensorSlotClaimReason::kSubscriberBorrowOrImport,
                         TensorSlotClaimReason::kDlpackOrFrameworkBorrow}));

  // Attach the receiving worker first, then let the producer write directly into its final
  // exportable loan and publish only the resulting descriptor.
  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  ASSERT_TRUE(CrossProcessTensorProbe::wait_for_sink_start());
  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  const bool started = CrossProcessTensorProbe::wait_for_starts();
  const bool received = started && CrossProcessTensorProbe::wait_for_sample();
  producer.request_stop();
  consumer.request_stop();
  producer.wait();
  consumer.wait();

  // Key assertion: the target observes the exact kValue bytes via a read-only imported lease
  // with kDLCPU device type, proving no copy or pointer sharing occurred.
  EXPECT_TRUE(received);
  EXPECT_EQ(CrossProcessTensorProbe::value.load(std::memory_order_acquire),
            CrossProcessTensorSource::kValue);
  EXPECT_TRUE(CrossProcessTensorProbe::read_only.load(std::memory_order_acquire));
  EXPECT_EQ(CrossProcessTensorProbe::device_type.load(std::memory_order_acquire), kDLCPU);
  EXPECT_EQ(producer.termination(), RunTermination::kStopped);
  EXPECT_EQ(consumer.termination(), RunTermination::kStopped);
  EXPECT_TRUE(producer.diagnostics().empty())
      << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
  EXPECT_TRUE(consumer.diagnostics().empty())
      << (consumer.diagnostics().empty() ? "" : consumer.diagnostics().front().token);
}

// A continuously clocked source must be able to cycle a finite direct-output export pool after the
// consumer takes and retires imported backing. Publishing more samples than the complete pool
// capacity proves forward progress rather than one cold fill.
//
//   source --[cross-process, host Tensor, depth=1]--> target
//
// REQ: CHAN-093
TEST(InterprocessPartitionBoundaryTest, DirectTensorPoolCyclesBeyondColdCapacity) {
  constexpr std::size_t kSampleCount = 8U;
  CrossProcessTensorProbe::reset();
  Graph graph{unique_runtime_graph_name("cross-process-direct-tensor-pool-cycle")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<CrossProcessTensorBurstSource>("source");
  auto target = graph.op<CrossProcessTensorSink>("target");
  graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 1U});

  // Key setup: the one-millisecond source emits twice the corrected four-slot pool capacity.
  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, split_source_target_placement());
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  ASSERT_EQ(state->tensor_allocations.size(), 1U);
  EXPECT_EQ(state->tensor_allocations.front().slot_count, 4U);
  const auto& service = state->cross_process_channel_services.front();

  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  ASSERT_TRUE(CrossProcessTensorProbe::wait_for_sink_start());
  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  const bool delivered = CrossProcessTensorProbe::wait_for_sample_count(kSampleCount);
  producer.request_stop();
  consumer.request_stop();
  producer.wait();
  consumer.wait();

  // Key assertion: every ordered sample crosses and retires, allowing the fixed pool to cycle.
  EXPECT_TRUE(delivered);
  EXPECT_EQ(CrossProcessTensorProbe::count.load(std::memory_order_acquire), kSampleCount);
  EXPECT_TRUE(CrossProcessTensorProbe::ordered.load(std::memory_order_acquire));
  EXPECT_EQ(CrossProcessTensorProbe::value.load(std::memory_order_acquire),
            CrossProcessTensorBurstSource::kValue + kSampleCount - 1U);
  EXPECT_EQ(producer.termination(), RunTermination::kStopped);
  EXPECT_EQ(consumer.termination(), RunTermination::kStopped);
  EXPECT_TRUE(producer.diagnostics().empty())
      << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
  EXPECT_TRUE(consumer.diagnostics().empty())
      << (consumer.diagnostics().empty() ? "" : consumer.diagnostics().front().token);
}

// Pinned host storage still carries a private CUDA synchronization device even though its public
// DLPack identity is kDLCUDAHost with device zero. The two workers must share only the HoloMQ
// handle, not a native cudaMallocHost pointer.
//
//   source --[cross-process, pinned Tensor, depth=1]--> target
//
// REQ: CHAN-093
TEST(InterprocessPartitionBoundaryTest, DirectPinnedTensorLoanCrossesProcessGroupsWithoutCopy) {
  CrossProcessTensorProbe::reset();
  Graph graph{unique_runtime_graph_name("cross-process-direct-pinned-tensor")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<CrossProcessPinnedTensorSource>("source");
  auto target = graph.op<CrossProcessPinnedTensorSink>("target");
  graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 1U});

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto output =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput && port.name == "output";
      });
  ASSERT_NE(output, snapshot->ports.end());
  const std::array device_bindings{
      detail::TensorDeviceBindingRecord{.port = PortId{output->id.value}, .device = DeviceId{0}},
  };

  // Key setup: device_bindings freezes device zero for the pinned output port so the
  // compiler can record the generation-sync device at compile time rather than at runtime.
  const ExecutionPlan plan = detail::compile_with_runtime_evidence_and_process_placement(
      graph,
      CompileOptions{},
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .tensor_device_bindings = device_bindings,
      },
      split_source_target_placement());
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(service.tensor_device, DeviceId{0});
  ASSERT_EQ(state->tensor_allocations.size(), 1U);
  EXPECT_EQ(service.tensor_pool_id, state->tensor_allocations.front().pool_id);
  EXPECT_EQ(state->tensor_allocations.front().memory_kind, MemoryKind::kPinnedHost);
  EXPECT_EQ(state->tensor_allocations.front().device, DeviceId{0});

  const auto target_resource =
      std::find_if(state->tensor_device_resources.begin(),
                   state->tensor_device_resources.end(),
                   [&](const auto& resource) {
                     return resource.process_group == service.target_process_group &&
                            resource.device == service.tensor_device;
                   });
  ASSERT_NE(target_resource, state->tensor_device_resources.end());
  EXPECT_EQ(target_resource->pool_count, 0U);
  EXPECT_EQ(target_resource->slot_capacity, 0U);

  // Start the importer first, then write directly into the final portable pinned SHM slot.
  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  ASSERT_TRUE(CrossProcessTensorProbe::wait_for_sink_start());
  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  const bool received = CrossProcessTensorProbe::wait_for_sample();
  producer.request_stop();
  consumer.request_stop();
  producer.wait();
  consumer.wait();

  // Key assertion: correct kValue bytes arrive with read_only=true and kDLCUDAHost device type,
  // confirming the HoloMQ handle (not a cudaMallocHost pointer) crossed the process boundary.
  EXPECT_TRUE(received);
  EXPECT_EQ(CrossProcessTensorProbe::value.load(std::memory_order_acquire),
            CrossProcessPinnedTensorSource::kValue);
  EXPECT_TRUE(CrossProcessTensorProbe::read_only.load(std::memory_order_acquire));
  EXPECT_EQ(CrossProcessTensorProbe::device_type.load(std::memory_order_acquire), kDLCUDAHost);
  EXPECT_EQ(producer.termination(), RunTermination::kStopped);
  EXPECT_EQ(consumer.termination(), RunTermination::kStopped);
  EXPECT_TRUE(producer.diagnostics().empty())
      << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
  EXPECT_TRUE(consumer.diagnostics().empty())
      << (consumer.diagnostics().empty() ? "" : consumer.diagnostics().front().token);
}

// The CUDA variant proves the producer's compute stream records the HoloMQ generation fence on
// the final device allocation and the receiving worker imports that allocation read-only.
//
//   source --[cross-process, CUDA device Tensor, depth=1]--> target
//
// REQ: CHAN-093
TEST(InterprocessPartitionBoundaryTest, DirectCudaTensorLoanPreservesDeviceOrderingAcrossGroups) {
  CrossProcessDeviceTensorProbe::reset();
  Graph graph{unique_runtime_graph_name("cross-process-direct-cuda-tensor")};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  auto source = graph.op<CrossProcessDeviceTensorSource>("source");
  auto target = graph.op<CrossProcessDeviceTensorSink>("target");
  graph.add_flow(source->output, target->input, ConnectionOptions{.queue_depth = 1U});

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(graph);
  const auto output =
      std::find_if(snapshot->ports.begin(), snapshot->ports.end(), [](const auto& port) {
        return port.direction == PortDirection::kOutput && port.name == "output";
      });
  ASSERT_NE(output, snapshot->ports.end());
  const std::array device_bindings{
      detail::TensorDeviceBindingRecord{.port = PortId{output->id.value}, .device = DeviceId{0}},
  };

  // Key setup: device_bindings freezes device zero for the CUDA output port; no
  // application-visible API or receiver-side inference selects a device at startup.
  const ExecutionPlan plan = detail::compile_with_runtime_evidence_and_process_placement(
      graph,
      CompileOptions{},
      detail::RuntimeCompileEvidenceView{
          .package_capabilities = detail::runtime_package_capability_view(),
          .tensor_device_bindings = device_bindings,
      },
      split_source_target_placement());
  ASSERT_TRUE(plan.ok()) << plan.json();
  const auto* state = plan_state(plan);
  ASSERT_NE(state, nullptr);
  ASSERT_EQ(state->cross_process_channel_services.size(), 1U);
  const auto& service = state->cross_process_channel_services.front();
  EXPECT_EQ(service.tensor_memory_kind, MemoryKind::kCudaDevice);
  EXPECT_EQ(service.tensor_device, DeviceId{0});
  const auto target_resource =
      std::find_if(state->tensor_device_resources.begin(),
                   state->tensor_device_resources.end(),
                   [&](const auto& resource) {
                     return resource.process_group == service.target_process_group &&
                            resource.device == service.tensor_device;
                   });
  ASSERT_NE(target_resource, state->tensor_device_resources.end());
  EXPECT_EQ(target_resource->pool_count, 0U);
  EXPECT_EQ(target_resource->slot_capacity, 0U);
  EXPECT_FALSE(target_resource->stream_id.empty());

  // Key assertion: kValue bytes arrive read-only after the sink consumes the generation fence,
  // proving device ordering crossed the process boundary without staging the payload.
  // The producer writes directly into its exportable device loan. The sink's blocking host probe
  // consumes the imported generation fence before checking bytes; the data itself is never staged.
  RunSession consumer = detail::run_async_worker(plan, service.target_process_group);
  ASSERT_TRUE(CrossProcessDeviceTensorProbe::wait_for_sink_start());
  RunSession producer = detail::run_async_worker(plan, service.source_process_group);
  const bool received = CrossProcessDeviceTensorProbe::wait_for_sample();
  producer.request_stop();
  consumer.request_stop();
  producer.wait();
  consumer.wait();

  EXPECT_TRUE(received);
  EXPECT_EQ(CrossProcessDeviceTensorProbe::value.load(std::memory_order_acquire),
            CrossProcessDeviceTensorSource::kValue);
  EXPECT_TRUE(CrossProcessDeviceTensorProbe::read_only.load(std::memory_order_acquire));
  EXPECT_EQ(producer.termination(), RunTermination::kStopped);
  EXPECT_EQ(consumer.termination(), RunTermination::kStopped);
  EXPECT_TRUE(producer.diagnostics().empty())
      << (producer.diagnostics().empty() ? "" : producer.diagnostics().front().token);
  EXPECT_TRUE(consumer.diagnostics().empty())
      << (consumer.diagnostics().empty() ? "" : consumer.diagnostics().front().token);
}

}  // namespace holoscan::test
