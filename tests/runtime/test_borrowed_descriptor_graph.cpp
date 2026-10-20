/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <chrono>
#include <memory>

#include "support/borrowed_descriptor_graph.hpp"

namespace holoscan::test::borrowed_descriptor {
namespace {

// REQ: OPERATOR-056
TEST(BorrowedDescriptorGraph, AcknowledgmentRequiresSinkPartition) {
  for (const bool separate_partition : {false, true}) {
    SCOPED_TRACE(separate_partition);
    auto probe = std::make_shared<Probe>(1);
    Graph graph{"retirement-partition-check"};
    auto source = graph.op<Source>("source", probe);
    auto sink = graph.op<Sink>("sink", probe, Mode::kBorrow, std::size_t{0});
    auto acknowledgment =
        graph.op<RetirementOrderedAcknowledgment<Probe>>("ack", probe, std::size_t{0});
    graph.add_flow(source->output, sink->input, ConnectionOptions{.queue_depth = 1});
    graph.add_flow(
        sink->acknowledgment, acknowledgment->input, ConnectionOptions{.queue_depth = 1});
    graph.partition("source").add(source);
    auto sink_partition = graph.partition("sink");
    sink_partition.add(sink);
    if (separate_partition)
      graph.partition("ack").add(acknowledgment);
    else
      sink_partition.add(acknowledgment);
    graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
    const auto plan = compile(graph);
    ASSERT_TRUE(plan.ok()) << plan.json();
    const auto partitions = plan.partitions();
    if (separate_partition) {
      EXPECT_THROW(require_retirement_ordered_acknowledgment(partitions.of(sink),
                                                             partitions.of(acknowledgment)),
                   RuntimeError);
    } else {
      EXPECT_NO_THROW(require_retirement_ordered_acknowledgment(partitions.of(sink),
                                                                partitions.of(acknowledgment)));
    }
  }
}

// REQ: OPERATOR-056
TEST(BorrowedDescriptorGraph, GateTimeoutExplainsMissingTestRelease) {
  AcknowledgmentRetirementGate gate;
  const auto held = gate.hold(std::chrono::milliseconds{0});
  ASSERT_FALSE(held);
  EXPECT_EQ(held.error().code, ErrorCode::kTimeout);
  EXPECT_EQ(held.error().message,
            "Retirement acknowledgment gate timed out waiting for test release");
  gate.release();
  EXPECT_TRUE(gate.hold(std::chrono::milliseconds{0}));
}

// REQ: OPERATOR-056
// REQ: SENSOR-IO-011
TEST(BorrowedDescriptorGraph, OwningAndBorrowedTransformsPreserveValuesAndBoundedProgress) {
  for (const Mode mode : {Mode::kOwning, Mode::kBorrow, Mode::kEdit}) {
    for (const bool force_transport : {false, true}) {
      for (const std::size_t sinks : {1U, 2U}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SCOPED_TRACE(force_transport);
        SCOPED_TRACE(sinks);
        auto probe = std::make_shared<Probe>(64, sinks);
        const auto plan = make_plan(probe, mode, force_transport);
        ASSERT_TRUE(plan.ok()) << plan.json();
        const auto* compiled = detail::ExecutionPlanAccess::state(plan);
        ASSERT_NE(compiled, nullptr);
        std::size_t direct_edges{};
        for (const auto& service : compiled->process_local_channel_services) {
          direct_edges +=
              service.delivery_kind == detail::ProcessLocalChannelDeliveryKind::kDirectRetained
                  ? 1U
                  : 0U;
        }
        // Data edges remain unchanged; each sink additionally publishes one scalar acknowledgment.
        EXPECT_EQ(direct_edges, force_transport ? 0U : (sinks == 1 ? 2U : 1U) + sinks);
        EXPECT_EQ(plan.execution_partitions().size(), 2U + sinks);
        for (const auto& allocation : plan.tensor_allocations()) {
          EXPECT_EQ(allocation.max_emits_per_compute, 1U);
          EXPECT_EQ(allocation.max_outstanding_loans, 1U);
        }
        auto session = run_async(plan);
        const bool completed = probe->wait();
        session.request_stop();
        session.wait();
        ASSERT_TRUE(completed) << (session.diagnostics().empty()
                                       ? "timeout"
                                       : session.diagnostics().front().token);
        EXPECT_EQ(session.termination(), RunTermination::kStopped);
        EXPECT_TRUE(probe->valid.load());
        EXPECT_EQ(probe->calls, 64U);
        EXPECT_EQ(probe->metadata_allocations, 0U);
        if (mode == Mode::kOwning) {
          EXPECT_GT(probe->receive_allocations, 0U);
        } else {
          EXPECT_EQ(probe->receive_allocations, 0U);
        }
      }
    }
  }
}

// REQ: OPERATOR-056
// REQ: SENSOR-IO-011
TEST(BorrowedDescriptorGraph, SourceCreditWaitsForSinkSelectionRetirement) {
  for (const Mode mode : {Mode::kOwning, Mode::kBorrow, Mode::kEdit}) {
    for (const bool force_transport : {false, true}) {
      for (const std::size_t sinks : {1U, 2U}) {
        SCOPED_TRACE(static_cast<int>(mode));
        SCOPED_TRACE(force_transport);
        SCOPED_TRACE(sinks);
        auto probe = std::make_shared<Probe>(8, sinks);
        probe->acknowledgment_gate = std::make_shared<AcknowledgmentRetirementGate>();
        const auto plan = make_plan(probe, mode, force_transport);
        ASSERT_TRUE(plan.ok()) << plan.json();
        EXPECT_EQ(plan.execution_partitions().size(), 2U + sinks);
        auto session = run_async(plan);
        const bool held = probe->acknowledgment_gate->wait_until_held();
        // The first sink has published its control message but still owns the selected input.
        // Checking credit under this explicit gate does not depend on a sleep or scheduler luck.
        EXPECT_TRUE(held)
            << "Timed out waiting for the sink to hold the retirement acknowledgment gate";
        EXPECT_EQ(probe->acknowledged[0].load(std::memory_order_acquire), 0U);
        probe->acknowledgment_gate->release();
        const bool completed = probe->wait();
        session.request_stop();
        session.wait();
        ASSERT_TRUE(completed);
        EXPECT_EQ(session.termination(), RunTermination::kStopped);
        EXPECT_TRUE(probe->valid.load());
        EXPECT_EQ(probe->calls, 8U);
      }
    }
  }
}

// REQ: SENSOR-IO-011
TEST(BorrowedDescriptorGraph, RejectedMetadataAndPublicationDoNotPinInputOrConsumeSequence) {
  for (const bool force_transport : {false, true}) {
    auto probe = std::make_shared<Probe>(32);
    const auto plan = make_plan(probe, Mode::kEdit, force_transport, true);
    ASSERT_TRUE(plan.ok()) << plan.json();
    auto session = run_async(plan);
    const bool completed = probe->wait();
    session.request_stop();
    session.wait();
    ASSERT_TRUE(completed) << (session.diagnostics().empty() ? "timeout"
                                                             : session.diagnostics().front().token);
    EXPECT_EQ(session.termination(), RunTermination::kStopped);
    EXPECT_TRUE(probe->valid.load());
    EXPECT_EQ(probe->calls, 32U);
    EXPECT_EQ(probe->receive_allocations, 0U);
    EXPECT_EQ(probe->metadata_allocations, 0U);
  }
}

}  // namespace
}  // namespace holoscan::test::borrowed_descriptor
