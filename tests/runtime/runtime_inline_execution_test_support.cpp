/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <sched.h>
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/run.hpp>
#include "compiler/execution_plan_state.hpp"
#include "core/domain/tensor_access.hpp"
#include "runtime/run_session_internal.hpp"
#include "runtime/runtime_diagnostic_sink.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime/runtime_test_access.hpp"
#include "runtime_inline_execution_test_support.hpp"

namespace holoscan::test::inline_execution {

using namespace std::chrono_literals;

[[nodiscard]] std::uint64_t phase_count(const InlinePartitionRunEvidence& evidence,
                                        detail::RuntimePhaseCounter counter) {
  return evidence.counters[detail::RuntimePhaseCounters::index(counter)];
}

void expect_no_legacy_lane_handoffs(const detail::RuntimePhaseCounters::Snapshot& counters) {
  const auto count = [&](detail::RuntimePhaseCounter counter) {
    return counters[detail::RuntimePhaseCounters::index(counter)];
  };
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
  EXPECT_EQ(count(detail::RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
}

[[nodiscard]] InlinePartitionRunEvidence capture_run_evidence(
    const RunSession& session, bool observed,
    const std::shared_ptr<detail::RuntimePhaseCounters>& counters,
    const std::shared_ptr<InlinePartitionControllerProbe>& control) {
  return InlinePartitionRunEvidence{
      .observed = observed,
      .termination = session.termination(),
      .diagnostic_count = session.diagnostics().size(),
      .controller_thread = control->controller_thread(),
      .compute_records = InlinePartitionProbeState::compute_records(),
      .lifecycle_records = InlinePartitionProbeState::lifecycle_records(),
      .param_compute_records = InlinePartitionProbeState::param_compute_records(),
      .param_lifecycle_records = InlinePartitionProbeState::param_lifecycle_records(),
      .temporal_predicate_records = InlinePartitionProbeState::temporal_predicate_records(),
      .sink_values = InlinePartitionProbeState::sink_values(),
      .schema_value = InlinePartitionProbeState::schema_value(),
      .counters = counters->snapshot(),
  };
}

[[nodiscard]] InlinePartitionRunEvidence run_and_observe(ExecutionPlan plan, std::size_t sink_count,
                                                         bool schema_sink) {
  const auto counters = std::make_shared<detail::RuntimePhaseCounters>();
  const auto control = std::make_shared<InlinePartitionControllerProbe>();
  plan = detail::RuntimeTestPlanAccess::with_phase_counters(plan, counters);
  plan = detail::RuntimeTestPlanAccess::with_support_thread_control(plan, control);

  RunSession session = run_async(plan);
  const bool observed = schema_sink ? InlinePartitionProbeState::wait_for_schema()
                                    : InlinePartitionProbeState::wait_for_sinks(sink_count);
  session.request_stop();
  session.wait();
  return capture_run_evidence(session, observed, counters, control);
}

[[nodiscard]] bool wait_for_terminal(const RunSession& session) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (session.termination() == RunTermination::kRunning &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return session.termination() != RunTermination::kRunning;
}

[[nodiscard]] ConnectionOptions queue_options(QueuePolicy policy) {
  return ConnectionOptions{
      .queue_policy = policy,
      .queue_depth = 2U,
  };
}

[[nodiscard]] ExecutionPlan make_pod_chain_plan(QueuePolicy policy, bool explicit_placement,
                                                std::size_t emission_limit,
                                                std::chrono::nanoseconds period) {
  // [source] -> [first relay] -> [second relay] -> [sink]
  Graph graph{explicit_placement ? "inline-pod-chain-fallback" : "inline-pod-chain"};
  auto source = graph.op<InlinePodSource>("source", 1U, 41U, false, emission_limit, period);
  auto first = graph.op<InlinePodRelay>("first-relay", 2U, 1U);
  auto second = graph.op<InlinePodRelay>("second-relay", 3U, 2U);
  auto sink = graph.op<InlinePodSink>("sink", 4U);
  graph.add_flow(source->output, first->input, queue_options(policy));
  graph.add_flow(first->output, second->input, queue_options(policy));
  graph.add_flow(second->output, sink->input, queue_options(policy));
  graph.set_default_clock(graph.add_clock<RealtimeClock>("inline-clock"));
  CompileOptions options;
  if (explicit_placement) {
    auto partition = graph.partition("inline-partition-fallback-worker");
    partition.add(source).add(first).add(second).add(sink);
    options.deployment.bind_thread_placement("inline-partition-fallback-worker", ThreadPlacement{});
  }
  return compile(graph, std::move(options));
}

[[nodiscard]] LifecycleTransactionOptions inline_lifecycle_capacity() {
  return LifecycleTransactionOptions{
      .max_pending_requests = 4U,
      .max_pending_transactions = 4U,
      .max_receipts = 8U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 64U,
      .max_audit_records = 8U,
      .max_quiesce_records = 8U,
      .max_rollback_records = 8U,
  };
}

void expect_inline_thread_evidence(const InlinePartitionRunEvidence& evidence,
                                   std::size_t expected_compute_count,
                                   std::size_t expected_operator_count,
                                   std::optional<std::size_t> expected_lifecycle_count) {
  const std::size_t lifecycle_count =
      expected_lifecycle_count.value_or(expected_operator_count * 2U);
  ASSERT_TRUE(evidence.controller_thread.has_value());
  ASSERT_EQ(evidence.compute_records.size(), expected_compute_count);
  ASSERT_EQ(evidence.lifecycle_records.size(), lifecycle_count);
  EXPECT_TRUE(std::ranges::all_of(evidence.compute_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_TRUE(std::ranges::all_of(evidence.lifecycle_records, [&](const auto& record) {
    return record.thread == *evidence.controller_thread;
  }));
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneComputeDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneDispatcherDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneLifecycleDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneUnknownDispatches), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneSleepingWakeSignals), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneBusyTaskPublications), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneCompletionsObserved), 0U);
  EXPECT_EQ(phase_count(evidence, detail::RuntimePhaseCounter::kLaneInlineOwnerInvocations),
            expected_compute_count + lifecycle_count +
                phase_count(evidence, detail::RuntimePhaseCounter::kReadinessPollDispatches));
}

}  // namespace holoscan::test::inline_execution
