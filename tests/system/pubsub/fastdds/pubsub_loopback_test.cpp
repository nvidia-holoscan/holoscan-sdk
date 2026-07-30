// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// @file
/// Single-process loopback test: TxOp -> PubSub -> RxOp within a single fragment.
///
/// This verifies that PubSub connectors can carry data between operators inside
/// one Holoscan application, exercising:
///   - Topic-based connection (no add_flow required; operators added individually).
///   - Backend extensibility: overriding Fragment::create_pubsub_network_context()
///     to provide FastDdsPubSubNetworkContext (the same pattern a third-party backend would use).
///   - FastDdsSerializer round-trip for scalar values.
///   - DDS discovery (writer/reader matching) within a single participant.
///   - QoS profiles set via IOSpec::qos() (backend-independent).
///
/// The test is parameterized over (scheduler_type, qos_label) tuples to verify
/// that message delivery works across scheduler and QoS combinations.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include <gxf/pubsub/qos_profile.hpp>
#include <holoscan/holoscan.hpp>
#include "holoscan/pubsub/fastdds/network_contexts/gxf/fastdds_pubsub_network_context.hpp"

namespace {

//==============================================================================
// Test parameter: scheduler type + QoS label
//==============================================================================

struct TestParam {
  std::string scheduler;
  std::string qos_label;  // "reliable_transient_local" (current test matrix)

  /// Resolve the qos_label to a QoSProfile, or nullopt for "default" (no explicit QoS).
  std::optional<nvidia::gxf::QoSProfile> qos_profile() const {
    if (qos_label == "reliable_transient_local") {
      return nvidia::gxf::QoSProfile::Default()
          .set_reliability(nvidia::gxf::ReliabilityPolicy::kReliable)
          .set_durability(nvidia::gxf::DurabilityPolicy::kTransientLocal);
    }
    // "default" — no explicit QoS; the GXF default (best-effort, volatile) is used.
    return std::nullopt;
  }
};

/// Pretty-print for gtest output: "greedy_default", "event_based_reliable_transient_local", etc.
std::string ParamName(const ::testing::TestParamInfo<TestParam>& info) {
  return info.param.scheduler + "_" + info.param.qos_label;
}

//==============================================================================
// Tx/Rx operators with optional QoS
//==============================================================================

/// Transmit operator that emits integer values over a PubSub connector.
/// Accepts an optional QoS profile set via IOSpec::qos().
class PingPubSubTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingPubSubTxOp)
  PingPubSubTxOp() = default;

  void set_qos(const nvidia::gxf::QoSProfile& qos) { qos_ = qos; }

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<int>("tick");
    auto& port = spec.output<int>("out").topic("ping");
    if (qos_.has_value()) {
      port.qos(qos_.value());
    }
    spec.param(target_count_,
               "target_count",
               "Target Count",
               "Number of messages to send after subscriber match",
               int64_t(10));
  }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto tick = op_input.receive<int>("tick");
    if (!tick) {
      return;
    }

    if (sent_count_ >= target_count_.get()) {
      return;
    }

    int value = ++index_;
    HOLOSCAN_LOG_INFO("PingPubSubTxOp: sending {}", value);
    op_output.emit(value, "out");
    ++sent_count_;
  }

 private:
  int index_ = 0;
  holoscan::Parameter<int64_t> target_count_;
  std::optional<nvidia::gxf::QoSProfile> qos_;
  int64_t sent_count_ = 0;
};

/// Receive operator that consumes integer values from a PubSub connector.
/// Accepts an optional QoS profile set via IOSpec::qos().
class PingPubSubRxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PingPubSubRxOp)
  PingPubSubRxOp() = default;

  void set_qos(const nvidia::gxf::QoSProfile& qos) { qos_ = qos; }

  void setup(holoscan::OperatorSpec& spec) override {
    auto& port = spec.input<int>("in").topic("ping");
    if (qos_.has_value()) {
      port.qos(qos_.value());
    }
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value = op_input.receive<int>("in");
    if (value) {
      HOLOSCAN_LOG_INFO("PingPubSubRxOp: received {}", value.value());
      ++received_count_;
    }
  }

  int received_count() const { return received_count_; }

 private:
  int received_count_ = 0;
  std::optional<nvidia::gxf::QoSProfile> qos_;
};

class TickOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TickOp)
  TickOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.output<int>("out"); }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    op_output.emit(++tick_index_, "out");
  }

 private:
  int tick_index_ = 0;
};

//==============================================================================
// Application
//==============================================================================

class PubSubLoopbackApp : public holoscan::Application {
 public:
  explicit PubSubLoopbackApp(const TestParam& param) : param_(param) {}

  // Override the virtual factory to explicitly provide the DDS backend.
  // This exercises the same extensibility pattern that a third-party backend
  // (e.g. Zenoh, UCX) would use — the base PubSubContext has no backend by
  // default, so the Application must provide one via this override.
  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    return make_network_context<holoscan::FastDdsPubSubNetworkContext>("pubsub_context");
  }

  void compose() override {
    using namespace holoscan;

    // All schedulers need stop_on_deadlock_timeout because DDS discovery/matching
    // is asynchronous: without a timeout the scheduler may declare deadlock before
    // the first message arrives.
    const int64_t deadlock_timeout_ms = 2000;

    if (param_.scheduler == "greedy") {
      scheduler(make_scheduler<GreedyScheduler>(
          "scheduler", Arg("stop_on_deadlock_timeout", deadlock_timeout_ms)));
    } else if (param_.scheduler == "event_based") {
      scheduler(make_scheduler<EventBasedScheduler>(
          "scheduler",
          Arg("worker_thread_number", static_cast<int64_t>(2)),
          Arg("stop_on_deadlock_timeout", deadlock_timeout_ms)));
    } else if (param_.scheduler == "multi_thread") {
      scheduler(make_scheduler<MultiThreadScheduler>(
          "scheduler",
          Arg("worker_thread_number", static_cast<int64_t>(2)),
          Arg("stop_on_deadlock_timeout", deadlock_timeout_ms)));
    } else {
      throw std::invalid_argument("Unknown scheduler type: " + param_.scheduler);
    }

    // Keep a short readiness window so total runtime stays near deadlock timeout.
    const int64_t tick_count = std::max<int64_t>(kMessageCount + 200, 300);
    auto tick = make_operator<TickOp>(
        "tick",
        make_condition<CountCondition>("tick_count", tick_count),
        make_condition<PeriodicCondition>("tick_period", std::chrono::milliseconds(1)));

    auto tx_ready = make_condition<SubscriberAvailableCondition>(
        "tx_ready",
        Arg("transmitter", "out"),
        Arg("min_subscriber_count", static_cast<uint64_t>(1)));
    tx_ = make_operator<PingPubSubTxOp>(
        "tx", tx_ready, Arg("target_count", static_cast<int64_t>(kMessageCount)));
    rx_ = make_operator<PingPubSubRxOp>("rx");

    // Apply QoS if specified in the test parameter.
    auto qos = param_.qos_profile();
    if (qos.has_value()) {
      tx_->set_qos(qos.value());
      rx_->set_qos(qos.value());
    }

    // Tick -> Tx provides a readiness barrier poll loop.
    // add_flow implicitly adds both tick and tx_ to the graph.
    add_flow(tick, tx_, {{"out", "tick"}});

    // PubSub operators connect by topic — no add_flow required between tx and rx.
    add_operator(rx_);
  }

  std::shared_ptr<PingPubSubRxOp> rx() const { return rx_; }

  static constexpr int kMessageCount = 100;

 private:
  TestParam param_;
  std::shared_ptr<PingPubSubTxOp> tx_;
  std::shared_ptr<PingPubSubRxOp> rx_;
};

//==============================================================================
// Parameterized test fixture
//==============================================================================

class PubSubLoopbackTest : public ::testing::TestWithParam<TestParam> {};

TEST_P(PubSubLoopbackTest, TxRxSingleFragment) {
  const auto& param = GetParam();
  auto app = holoscan::make_application<PubSubLoopbackApp>(param);

  // Run the app: FastDdsPubSubNetworkContext is auto-created via create_pubsub_network_context().
  app->run();

  // Verify that the receiver actually received messages through the DDS pub/sub path.
  auto rx = app->rx();
  ASSERT_NE(rx, nullptr);

  const int expected = PubSubLoopbackApp::kMessageCount;
  const int actual = rx->received_count();

  if (param.qos_label == "default") {
    // Best-effort + volatile can drop some samples due to async matching/scheduling.
    EXPECT_GT(actual, 10) << "Expected more than 10 messages with default QoS, but received "
                          << actual << " (scheduler: " << param.scheduler
                          << ", qos: " << param.qos_label << ")";
  } else {
    // Greedy and EventBased schedulers should deliver all messages with reliable QoS.
    EXPECT_EQ(actual, expected) << "Expected " << expected
                                << " messages to be delivered via DDS pub/sub, but received "
                                << actual << " (scheduler: " << param.scheduler
                                << ", qos: " << param.qos_label << ")";
  }
}

// Test matrix: scheduler variants under deterministic QoS settings.
INSTANTIATE_TEST_SUITE_P(SchedulersAndQoS, PubSubLoopbackTest,
                         ::testing::Values(
                             // Default QoS (best-effort, volatile) — works for slow schedulers
                             // where DDS matching completes before messages are sent.
                             TestParam{"greedy", "default"}, TestParam{"event_based", "default"},
                             TestParam{"multi_thread", "default"},
                             // Reliable + transient-local QoS
                             TestParam{"greedy", "reliable_transient_local"},
                             TestParam{"event_based", "reliable_transient_local"},
                             TestParam{"multi_thread", "reliable_transient_local"}),
                         ParamName);

}  // namespace
