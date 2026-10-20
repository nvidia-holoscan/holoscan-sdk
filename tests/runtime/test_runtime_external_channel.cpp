/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <holomq/context_builder.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "../common/cuda_test_utils.hpp"
#include "image_generated.h"

#include "channel/channel_common.hpp"
#include "channel/channel_payload_frame.hpp"
#include "channel/channel_tensor_test_access.hpp"
#include "channel/endpoint_catalog_state.hpp"
#include "compiler/compile_internal.hpp"
#include "compiler/process_placement_internal.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/endpoint_clock_health_runtime.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_support_thread_control.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime/runtime_work_inventory.hpp"
#include "runtime/runtime_worker.hpp"
#include "runtime_external_channel_test_support.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan {
namespace {

using namespace std::chrono_literals;

using namespace test::external_channel;

// Records cold Runtime support-thread entry so external-import tests can distinguish the
// controller-owned catalog drain from the process-local HoloMQ pump.
class ExternalImportSupportThreadProbe final : public detail::RuntimeSupportThreadControl {
 public:
  [[nodiscard]] expected<void, Error> enter(const SupportThreadRole role) override {
    entered_[static_cast<std::size_t>(role)].store(true, std::memory_order_release);
    return {};
  }

  [[nodiscard]] bool entered(const SupportThreadRole role) const noexcept {
    return entered_[static_cast<std::size_t>(role)].load(std::memory_order_acquire);
  }

 private:
  std::array<std::atomic_bool, 2U> entered_{};
};

// Records whether compute was ever called on a graph where an external topic exists only
// in the catalog with no wired edge to any operator.
// Representative test: DisconnectedImportRemainsCatalogOnlyAtRuntime.
struct CatalogOnlyRuntimeProbe {
  static void reset() noexcept { compute_count.store(0U, std::memory_order_relaxed); }

  [[nodiscard]] static bool wait_for_compute(std::chrono::milliseconds timeout = 5s) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, timeout, [] { return compute_count.load(std::memory_order_acquire) != 0U; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::uint32_t> compute_count{};
};

// Clock-triggered operator with no inputs or outputs. Proves that a catalog-registered
// external topic does not block execution when it has no wired edges.
// Used by: DisconnectedImportRemainsCatalogOnlyAtRuntime,
//          DisconnectedExportRemainsCatalogOnlyAtRuntime.
class CatalogOnlyClockOperator final : public Operator<> {
 public:
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    CatalogOnlyRuntimeProbe::compute_count.fetch_add(1U, std::memory_order_release);
    CatalogOnlyRuntimeProbe::ready.notify_all();
    return {};
  }
};

struct CatalogScalarReuseProbe {
  static void reset() noexcept {
    const std::lock_guard lock{mutex};
    count.store(0U, std::memory_order_release);
    values.fill(0U);
    sequences.fill(0U);
    states.fill(InputState::kAbsent);
  }

  static void record(const Sample<std::uint32_t>& sample, const InputState state) noexcept {
    const std::lock_guard lock{mutex};
    const std::size_t index = count.load(std::memory_order_relaxed);
    if (index < values.size()) {
      values[index] = sample.data;
      sequences[index] = sample.metadata.sequence;
      states[index] = state;
    }
    count.store(index + 1U, std::memory_order_release);
    ready.notify_all();
  }

  [[nodiscard]] static bool wait_for_count(const std::size_t expected) {
    std::unique_lock lock{mutex};
    return ready.wait_for(
        lock, 5s, [expected] { return count.load(std::memory_order_acquire) >= expected; });
  }

  static inline std::mutex mutex;
  static inline std::condition_variable ready;
  static inline std::atomic<std::size_t> count{};
  static inline std::array<std::uint32_t, 2U> values{};
  static inline std::array<std::uint64_t, 2U> sequences{};
  static inline std::array<InputState, 2U> states{};
};

class CatalogScalarReuseSink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.input(input, "input").queue_depth(1U); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnAnyInput{})
        .input(input, LatestWithin{.max_age = 30s})
        .on_missing(input, ReuseLast{.max_age = 30s});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    CatalogScalarReuseProbe::record(*sample, input.selection().state);
    return {};
  }

  Input<std::uint32_t> input;
};

[[nodiscard]] detail::ProcessPlacementBinding external_endpoint_process_placement(
    std::string_view endpoint_uri) {
  const std::array selectors{detail::GraphObjectSelector{.external_topic = endpoint_uri}};
  detail::ProcessPlacementBinding placement;
  placement.bind_process_role("external-endpoint",
                              detail::ProcessPlacement{
                                  .isolation = detail::ProcessIsolation::kExternalProcess,
                                  .members = selectors,
                                  .launch_group = "external-endpoint",
                              });
  return placement;
}

// REQ: CHAN-073
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, DisconnectedImportRemainsCatalogOnlyAtRuntime) {
  CatalogOnlyRuntimeProbe::reset();
  //   holomq://endpoint  (catalog-only, no edge to any operator)
  //   independent        (OnClock period=30s, no inputs/outputs)
  Graph graph{"runtime-disconnected-external-import"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key setup: external topic registered in catalog but not connected to any operator.
  ASSERT_EQ(plan.graph().external_topics().size(), 1U);
  ASSERT_TRUE(plan.payload_edges().empty());

  RunSession session = run_async(plan);
  const bool computed = CatalogOnlyRuntimeProbe::wait_for_compute();
  session.request_stop();
  session.wait();

  // Key assertion: graph runs normally despite the catalog-only import topic.
  EXPECT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-073
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, RejectsExternalTopicWithoutEndpointCatalogAuthority) {
  // Graph topology:
  //   import topic    independent clock-triggered operator  (no connecting edge)
  Graph graph{"runtime-external-topic-without-catalog"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan compiled = compile(graph);
  ASSERT_TRUE(compiled.ok()) << compiled.json();
  ASSERT_EQ(compiled.graph().external_topics().size(), 1U);

  // Key setup: retain the authored topic while removing its immutable catalog authority.
  const ExecutionPlan malformed = detail::RuntimeTestPlanAccess::remove_endpoint_catalog(compiled);
  ASSERT_TRUE(malformed.ok()) << malformed.json();
  const auto state = detail::ExecutionPlanAccess::retain(malformed);
  ASSERT_NE(state, nullptr);

  const auto reconstructed = detail::GraphExec::reconstruct(state, RunGenerationId{1U});

  // Key assertion: replay fails closed before consulting Runtime external bindings.
  ASSERT_FALSE(reconstructed);
  EXPECT_EQ(reconstructed.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(reconstructed.error().message, "RUNTIME_EXTERNAL_BINDING_UNAVAILABLE");
}

// REQ: CHAN-073
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, DisconnectedExportRemainsCatalogOnlyAtRuntime) {
  CatalogOnlyRuntimeProbe::reset();
  //   holomq://endpoint  (catalog-only, no edge from any operator)
  //   independent        (OnClock period=30s, no inputs/outputs)
  Graph graph{"runtime-disconnected-external-export"};
  (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key setup: external topic registered in catalog but not connected to any operator.
  ASSERT_EQ(plan.graph().external_topics().size(), 1U);
  ASSERT_TRUE(plan.payload_edges().empty());

  RunSession session = run_async(plan);
  const bool computed = CatalogOnlyRuntimeProbe::wait_for_compute();
  session.request_stop();
  session.wait();

  // Key assertion: graph runs normally despite the catalog-only export topic.
  EXPECT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// An endpoint-only include_all() plan retains one topology lane without inventing a callback
// owner. Runtime must accept that lane and complete the catalog-only application as a no-op.
// REQ: CHAN-073
// REQ: COMPILE-068
// REQ: GRAPH-032
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     IncludeAllStandaloneImportRetainsTopologyLaneWithoutCallbackOwner) {
  // Graph topology:
  //   standalone import topic  (included in the application partition)
  Graph graph{"runtime-include-all-standalone-import"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  graph.partition("application").include_all();

  // Key setup: include the endpoint-only topology in one authored partition.
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"application"});
  ASSERT_EQ(plan.execution_partitions().front().members.size(), 1U);
  EXPECT_TRUE(plan.execution_partitions().front().operator_paths.empty());
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.execution_lanes().front().partition, plan.execution_partitions().front().id);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 0U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 0U);
  EXPECT_TRUE(plan.payload_edges().empty());
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  EXPECT_TRUE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_FALSE(detail::has_plan_callback_owners(*accepted));
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_FALSE((*reconstructed)->has_process_local_callback_owners());

  RunSession session = run_async(plan);
  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: the structural lane has no callback owner and completes as a no-op.
  EXPECT_TRUE(completed);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-073
// REQ: COMPILE-068
// REQ: GRAPH-032
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     IncludeAllStandaloneExportRetainsTopologyLaneWithoutCallbackOwner) {
  // Graph topology:
  //   standalone export topic  (included in the application partition)
  Graph graph{"runtime-include-all-standalone-export"};
  (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  graph.partition("application").include_all();

  // Key setup: include the endpoint-only topology in one authored partition.
  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"application"});
  ASSERT_EQ(plan.execution_partitions().front().members.size(), 1U);
  EXPECT_TRUE(plan.execution_partitions().front().operator_paths.empty());
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.execution_lanes().front().partition, plan.execution_partitions().front().id);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 0U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 0U);
  EXPECT_TRUE(plan.payload_edges().empty());
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto accepted = detail::ExecutionPlanAccess::retain(plan);
  ASSERT_NE(accepted, nullptr);
  EXPECT_TRUE(detail::has_structural_runtime_topology(*accepted));
  EXPECT_FALSE(detail::has_plan_callback_owners(*accepted));
  auto reconstructed = detail::GraphExec::reconstruct(accepted, RunGenerationId{17U});
  ASSERT_TRUE(reconstructed) << reconstructed.error().message;
  EXPECT_FALSE((*reconstructed)->has_process_local_callback_owners());

  RunSession session = run_async(plan);
  const bool completed = wait_for_session_terminal(session);
  if (!completed) {
    session.request_stop();
  }
  session.wait();

  // Key assertion: the structural lane has no callback owner and completes as a no-op.
  EXPECT_TRUE(completed);
  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// Interface-only topology retains authored clocks as authenticated plan evidence, but no callback
// owner can observe a run-owned clock instance. Cover both external-topic directions.
//
// Graph topology (one case per direction):
//   imported-topic   or   exported-topic   (no Operator callback owner)
// REQ: RUNTIME-010
// REQ: RUNTIME-015
TEST(RuntimeExternalChannelTest, StandaloneExternalTopicsDoNotConstructUnobservableClocks) {
  for (const bool imported : {true, false}) {
    SCOPED_TRACE(imported ? "import" : "export");
    Graph graph{imported ? "runtime-standalone-import-clock" : "runtime-standalone-export-clock"};
    if (imported) {
      (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
    } else {
      (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
    }
    graph.partition("application").include_all();
    const ClockKey default_clock = graph.add_clock<test::RuntimeProbeClock>("runtime-probe-clock");
    graph.set_default_clock(default_clock);
    const ExecutionPlan plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    ASSERT_FALSE(plan.graph().vertices().empty());
    ASSERT_TRUE(std::ranges::none_of(plan.graph().vertices(), [](const auto& vertex) {
      return vertex.kind == VertexKind::kOperator;
    }));

    // Key setup: exclude compile-time factory activity from the Runtime observation.
    test::RuntimeProbeState::reset();
    RunSession session = run_async(plan);
    const bool completed = wait_for_session_terminal(session);
    if (!completed) {
      session.request_stop();
    }
    session.wait();

    // Key assertion: the interface-only run completes without constructing an inert clock.
    EXPECT_TRUE(completed);
    EXPECT_EQ(session.termination(), RunTermination::kCompleted);
    EXPECT_TRUE(session.diagnostics().empty())
        << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
    EXPECT_EQ(test::RuntimeProbeState::clocks_constructed.load(std::memory_order_relaxed), 0U);
  }
}

// Disconnected catalog endpoints remain part of the exact include_all() executable cover while
// only the Operator-bearing lane contributes a Runtime callback owner.
// REQ: CHAN-073
// REQ: COMPILE-068
// REQ: GRAPH-032
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest, IncludeAllRunsMixedOperatorAndDisconnectedEndpointsUnderOneOwner) {
  CatalogOnlyRuntimeProbe::reset();
  Graph graph{"runtime-include-all-disconnected-mixed"};
  (void)graph.import_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.export_topic<std::uint32_t>("holomq://" + unique_endpoint_name());
  (void)graph.op<CatalogOnlyClockOperator>("independent");
  const ClockKey default_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(default_clock);
  graph.partition("application").include_all();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  EXPECT_EQ(plan.execution_partitions().front().authored_name,
            std::optional<std::string>{"application"});
  EXPECT_EQ(plan.execution_partitions().front().members.size(), 3U);
  EXPECT_EQ(plan.execution_partitions().front().operator_paths,
            std::vector<std::string>{"independent"});
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().max_concurrent_activations, 1U);
  EXPECT_FALSE(plan.scheduler_plan().activation_overlap);
  EXPECT_TRUE(plan.payload_edges().empty());
  EXPECT_TRUE(plan.schedule_edges().empty());

  RunSession session = run_async(plan);
  const bool computed = CatalogOnlyRuntimeProbe::wait_for_compute();
  session.request_stop();
  session.wait();

  EXPECT_TRUE(computed);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A constrained automatic cover attaches a sole adjacent imported endpoint to the authored
// Operator owner when both vertices share the default process group. The merged owner retains the
// external Channel receive boundary and executes the transferred sample.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest, SameProcessImportEndpointAttachesToAuthoredOwnerAndReceives) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-same-process-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});
  graph.partition("anchor").add(sink);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto& anchor = plan.execution_partitions().front();
  EXPECT_EQ(anchor.authored_name, std::optional<std::string>{"anchor"});
  EXPECT_EQ(anchor.members.size(), 2U);
  EXPECT_EQ(anchor.operator_paths, std::vector<std::string>{"sink"});
  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kImportedTopic; });
  const auto sink_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "sink";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(sink_vertex, plan.graph().vertices().end());
  EXPECT_NE(std::ranges::find(anchor.members, endpoint_vertex->id), anchor.members.end());
  EXPECT_NE(std::ranges::find(anchor.members, sink_vertex->id), anchor.members.end());
  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto sink_placement = std::ranges::find(
      placements, sink_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(sink_placement, placements.end());
  EXPECT_EQ(endpoint_placement->process_group, sink_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "same-process import subscriber did not connect: "
           << runtime_error_message(connected.error());
  }
  ASSERT_TRUE(publisher->emit_captured(0x12345678U, 100));
  const bool received = ExternalImportProbe::wait_for_count(1U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received);
  EXPECT_EQ(ExternalImportProbe::values[0], 0x12345678U);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 1U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// The export-side same-process case is symmetric: the exported endpoint joins the authored source
// owner, which retains and drives the external publisher without a crossing schedule edge.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest, SameProcessExportEndpointAttachesToAuthoredOwnerAndPublishes) {
  ExternalScalarExportSource::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-same-process-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  const auto exported = graph.export_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 4U});
  graph.partition("anchor").add(source);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = compile(graph);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 1U);
  ASSERT_EQ(plan.execution_lanes().size(), 1U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 1U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  EXPECT_TRUE(plan.schedule_edges().empty());
  const auto& anchor = plan.execution_partitions().front();
  EXPECT_EQ(anchor.authored_name, std::optional<std::string>{"anchor"});
  EXPECT_EQ(anchor.members.size(), 2U);
  EXPECT_EQ(anchor.operator_paths, std::vector<std::string>{"source"});
  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kExportedTopic; });
  const auto source_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "source";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(source_vertex, plan.graph().vertices().end());
  EXPECT_NE(std::ranges::find(anchor.members, endpoint_vertex->id), anchor.members.end());
  EXPECT_NE(std::ranges::find(anchor.members, source_vertex->id), anchor.members.end());
  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto source_placement = std::ranges::find(
      placements, source_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(source_placement, placements.end());
  EXPECT_EQ(endpoint_placement->process_group, source_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber =
      external_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = run_async(plan);
  if (!ExternalScalarExportSource::wait_for_compute_started()) {
    ExternalScalarExportSource::allow_emit();
    session.request_stop();
    session.wait();
    FAIL() << "same-process export source did not enter compute; termination="
           << static_cast<int>(session.termination());
  }
  auto publisher_ready = wait_for_publisher(*subscriber);
  ExternalScalarExportSource::allow_emit();
  ASSERT_TRUE(publisher_ready) << runtime_error_message(publisher_ready.error());
  auto received = wait_for_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 0xdecafbadU);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// A constrained automatic cover may attach an external import to its sole authored neighbor only
// when both vertices share one process. A separately placed endpoint retains its topology lane,
// and the Operator worker still owns the external-channel receive boundary.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     CrossProcessImportEndpointRemainsSeparateAndOperatorWorkerReceives) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-cross-process-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported = graph.import_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});
  graph.partition("anchor").add(sink);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, external_endpoint_process_placement(endpoint_uri));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_EQ(plan.execution_lanes().size(), 2U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 2U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  ASSERT_EQ(plan.schedule_edges().size(), 1U);

  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kImportedTopic; });
  const auto sink_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "sink";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(sink_vertex, plan.graph().vertices().end());
  const auto endpoint_partition =
      std::ranges::find_if(plan.execution_partitions(), [&](const auto& partition) {
        return partition.members == std::vector<VertexId>{endpoint_vertex->id};
      });
  const auto anchor_partition = std::ranges::find_if(
      plan.execution_partitions(),
      [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(endpoint_partition, plan.execution_partitions().end());
  ASSERT_NE(anchor_partition, plan.execution_partitions().end());
  EXPECT_NE(endpoint_partition->id, anchor_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().source, endpoint_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().destination, anchor_partition->id);

  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto sink_placement = std::ranges::find(
      placements, sink_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(sink_placement, placements.end());
  ASSERT_NE(endpoint_placement->process_group, sink_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = detail::run_async_worker(plan, sink_placement->process_group);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "cross-process import subscriber did not connect: "
           << runtime_error_message(connected.error());
  }
  ASSERT_TRUE(publisher->emit_captured(0x12345678U, 100));
  const bool received = ExternalImportProbe::wait_for_count(1U);
  session.request_stop();
  session.wait();

  ASSERT_TRUE(received);
  EXPECT_EQ(ExternalImportProbe::values[0], 0x12345678U);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 1U);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// The export-side process-boundary case is symmetric: the endpoint keeps its synthetic lane while
// the authored source worker materializes and drives the external publisher.
// REQ: COMPILE-068
// REQ: RUNTIME-015
// REQ: RUNTIME-041
TEST(RuntimeExternalChannelTest,
     CrossProcessExportEndpointRemainsSeparateAndOperatorWorkerPublishes) {
  ExternalScalarExportSource::reset();
  const std::string endpoint_name = unique_endpoint_name();
  const std::string endpoint_uri = "holomq://" + endpoint_name;

  Graph graph{"runtime-constrained-cross-process-export"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  const auto exported = graph.export_topic<std::uint32_t>(endpoint_uri, std::move(options));
  const auto source = graph.op<ExternalScalarExportSource>("source");
  graph.add_flow(source->output, exported.input(), ConnectionOptions{.queue_depth = 4U});
  graph.partition("anchor").add(source);
  graph.auto_partition_remaining();

  const ExecutionPlan plan = detail::compile_with_process_placement(
      graph, CompileOptions{}, external_endpoint_process_placement(endpoint_uri));
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.execution_partitions().size(), 2U);
  ASSERT_EQ(plan.execution_lanes().size(), 2U);
  EXPECT_EQ(plan.scheduler_plan().execution_lane_count, 2U);
  EXPECT_EQ(plan.scheduler_plan().worker_count, 1U);
  ASSERT_EQ(plan.schedule_edges().size(), 1U);

  const auto endpoint_vertex = std::ranges::find_if(
      plan.graph().vertices(),
      [](const auto& vertex) { return vertex.kind == VertexKind::kExportedTopic; });
  const auto source_vertex = std::ranges::find_if(plan.graph().vertices(), [](const auto& vertex) {
    return vertex.kind == VertexKind::kOperator && vertex.name == "source";
  });
  ASSERT_NE(endpoint_vertex, plan.graph().vertices().end());
  ASSERT_NE(source_vertex, plan.graph().vertices().end());
  const auto endpoint_partition =
      std::ranges::find_if(plan.execution_partitions(), [&](const auto& partition) {
        return partition.members == std::vector<VertexId>{endpoint_vertex->id};
      });
  const auto anchor_partition = std::ranges::find_if(
      plan.execution_partitions(),
      [](const auto& partition) { return partition.authored_name == "anchor"; });
  ASSERT_NE(endpoint_partition, plan.execution_partitions().end());
  ASSERT_NE(anchor_partition, plan.execution_partitions().end());
  EXPECT_NE(endpoint_partition->id, anchor_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().source, anchor_partition->id);
  EXPECT_EQ(plan.schedule_edges().front().destination, endpoint_partition->id);

  const auto placements = detail::ProcessPlacementPlanAccess::vertex_process_placements(plan);
  const auto endpoint_placement = std::ranges::find(
      placements, endpoint_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  const auto source_placement = std::ranges::find(
      placements, source_vertex->id, &detail::VertexProcessPlacementPlanRecord::vertex);
  ASSERT_NE(endpoint_placement, placements.end());
  ASSERT_NE(source_placement, placements.end());
  ASSERT_NE(endpoint_placement->process_group, source_placement->process_group);

  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto subscriber =
      external_runtime->create_subscriber<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(subscriber) << runtime_error_message(subscriber.error());

  RunSession session = detail::run_async_worker(plan, source_placement->process_group);
  if (!ExternalScalarExportSource::wait_for_compute_started()) {
    ExternalScalarExportSource::allow_emit();
    session.request_stop();
    session.wait();
    FAIL() << "cross-process export source did not enter compute; termination="
           << static_cast<int>(session.termination());
  }
  auto publisher_ready = wait_for_publisher(*subscriber);
  ExternalScalarExportSource::allow_emit();
  ASSERT_TRUE(publisher_ready) << runtime_error_message(publisher_ready.error());
  auto received = wait_for_sample(*subscriber);

  session.request_stop();
  session.wait();
  ASSERT_TRUE(received) << runtime_error_message(received.error());
  EXPECT_EQ(received->data, 0xdecafbadU);
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
}

// REQ: CHAN-069
// REQ: RUNTIME-011
// REQ: RUNTIME-016
// REQ: RUNTIME-017
TEST(RuntimeExternalChannelTest, UnmonitoredScalarImportDrainsEverySampleInPublisherOrder) {
  ExternalImportProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  //   external_publisher --(holomq://endpoint, unmonitored)--> imported_topic
  //       --[depth=8]--> sink
  Graph graph{"runtime-unmonitored-external-import"};
  const ClockKey capture_clock = graph.add_clock<ManualClock>("capture-clock", std::int64_t{1000});
  graph.set_default_clock(capture_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = capture_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<ExternalImportSink>("sink");
  graph.add_flow(imported.output(), sink->input, ConnectionOptions{.queue_depth = 8U});

  const ExecutionPlan compiled = compile(graph);
  auto support_threads = std::make_shared<ExternalImportSupportThreadProbe>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_support_thread_control(compiled, support_threads);
  ASSERT_TRUE(plan.ok()) << plan.json();
  // Key setup: import has RequireCaptureTime but no ClockHealthOptions. Runtime must drain this
  // externally driven catalog input on the controller path rather than starting a Channel pump.
  ASSERT_EQ(plan.endpoint_clock_health().size(), 1U);
  EXPECT_EQ(plan.endpoint_clock_health().front().capability, ClockHealthCapability::kUnavailable);
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "unmonitored external subscriber did not connect: "
           << runtime_error_message(connected.error());
  }

  constexpr std::array kValues{0x01020304U, 0x11223344U};
  for (std::size_t index = 0U; index < kValues.size(); ++index) {
    ASSERT_TRUE(publisher->emit_captured(kValues[index], 100 + static_cast<std::int64_t>(index)));
  }
  EXPECT_TRUE(ExternalImportProbe::wait_for_count(kValues.size()));

  session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(), RunTermination::kStopped);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  // Key assertion: both values arrive in emission order with monotonically increasing sequences.
  EXPECT_EQ(ExternalImportProbe::count.load(std::memory_order_acquire), kValues.size());
  EXPECT_EQ(ExternalImportProbe::values[0], kValues[0]);
  EXPECT_EQ(ExternalImportProbe::values[1], kValues[1]);
  EXPECT_EQ(ExternalImportProbe::sequences[0], 1U);
  EXPECT_EQ(ExternalImportProbe::sequences[1], 2U);
  EXPECT_TRUE(support_threads->entered(SupportThreadRole::kController));
  EXPECT_FALSE(support_threads->entered(SupportThreadRole::kChannelPump));
}

// The catalog inline wrapper may retain N+1 in its own bounded staging buffer while the typed
// depth-one slot still owns ReuseLast N. That wrapper-owned head must participate in the same
// controller handoff as raw Tensor and descriptor-schema imports before ordered EOS is consumed.
// REQ: CHAN-136
// REQ: RUNTIME-050
TEST(RuntimeExternalChannelTest, CatalogInlineReuseLastPublishesFinalDataBeforeOrderedTerminal) {
  CatalogScalarReuseProbe::reset();
  const std::string endpoint_name = unique_endpoint_name();

  Graph graph{"runtime-catalog-inline-reuse-terminal"};
  const ClockKey runtime_clock = graph.add_clock<RealtimeClock>("runtime-clock");
  graph.set_default_clock(runtime_clock);
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.capture_time = RequireCaptureTime{.clock = runtime_clock};
  const auto imported =
      graph.import_topic<std::uint32_t>("holomq://" + endpoint_name, std::move(options));
  const auto sink = graph.op<CatalogScalarReuseSink>("sink");
  graph.add_flow(imported.output(),
                 sink->input,
                 ConnectionOptions{.queue_policy = QueuePolicy::kDropOldest, .queue_depth = 1U});

  const ExecutionPlan compiled = compile(graph);
  auto phase_counters = std::make_shared<detail::RuntimePhaseCounters>();
  const ExecutionPlan plan =
      detail::RuntimeTestPlanAccess::with_phase_counters(compiled, phase_counters);
  ASSERT_TRUE(plan.ok()) << plan.json();
  auto external_runtime = ChannelRuntime::create();
  ASSERT_TRUE(external_runtime) << runtime_error_message(external_runtime.error());
  auto publisher =
      external_runtime->create_publisher<std::uint32_t>(plan.endpoint_catalog(), endpoint_name);
  ASSERT_TRUE(publisher) << runtime_error_message(publisher.error());

  RunSession session = run_async(plan);
  if (auto connected = wait_for_subscriber(*publisher); !connected) {
    session.request_stop();
    session.wait();
    FAIL() << "catalog inline subscriber did not connect: "
           << runtime_error_message(connected.error());
  }
  const std::int64_t capture = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count();
  ASSERT_TRUE(publisher->emit_captured(1U, capture));
  ASSERT_TRUE(CatalogScalarReuseProbe::wait_for_count(1U));
  ASSERT_TRUE(publisher->emit_captured(2U, capture + 1));
  ASSERT_TRUE(publisher->close());
  ASSERT_TRUE(CatalogScalarReuseProbe::wait_for_count(2U))
      << "evictions="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionEvictions)]
      << " readiness="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kChannelReadinessPublished)]
      << " backpressured="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kChannelReceiveBackpressured)]
      << " candidates="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionCandidates)]
      << " declined="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedAboveFloor)]
      << " refused="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedEvictionRefused)]
      << " not_superseded="
      << phase_counters->snapshot()[detail::RuntimePhaseCounters::index(
             detail::RuntimePhaseCounter::kTemporalRetentionDeclinedNotSuperseded)]
      << " running=" << session.is_running() << " diagnostics=" << session.diagnostics().size();
  session.wait();

  EXPECT_EQ(session.termination(), RunTermination::kCompleted);
  EXPECT_TRUE(session.diagnostics().empty())
      << (session.diagnostics().empty() ? "" : session.diagnostics().front().token);
  EXPECT_EQ(CatalogScalarReuseProbe::count.load(std::memory_order_acquire), 2U);
  EXPECT_EQ(CatalogScalarReuseProbe::values, (std::array<std::uint32_t, 2U>{1U, 2U}));
  EXPECT_EQ(CatalogScalarReuseProbe::sequences, (std::array<std::uint64_t, 2U>{1U, 2U}));
  EXPECT_EQ(CatalogScalarReuseProbe::states,
            (std::array<InputState, 2U>{InputState::kSelected, InputState::kSelected}));
}

}  // namespace
}  // namespace holoscan
