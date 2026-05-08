// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// @file
/// Single-process loopback test: TxOp → PubSub → RxOp within a single fragment,
/// using InMemoryPubSubNetworkContext.
///
/// Key differences from the FastDDS loopback test:
///   - No external discovery protocol: in-memory registration is synchronous.
///   - All messages are guaranteed to be delivered (kPassthrough mode, no drops).
///   - No async matching latency: deadlock timeout can be short.
///   - Tests both GreedyScheduler and EventBasedScheduler.

#include <gtest/gtest.h>

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/pubsub/in_memory/network_contexts/gxf/in_memory_pubsub_network_context.hpp>

namespace {

//==============================================================================
// Test operators
//==============================================================================

/// Emits integer values over a pub/sub topic.
/// Stops after message_count values have been sent.
class TxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TxOp)
  TxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<int>("out").topic("ping");
    spec.param(message_count_,
               "message_count",
               "Message Count",
               "Number of messages to publish",
               int{100});
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    if (sent_ < message_count_.get()) {
      ++sent_;
      HOLOSCAN_LOG_INFO("[TxOp] compute #{}: emitting value {}", compute_count_, sent_);
      op_output.emit(sent_, "out");
    } else {
      HOLOSCAN_LOG_INFO("[TxOp] compute #{}: all {} sent, no-op", compute_count_, sent_);
    }
    ++compute_count_;
  }

  int sent() const { return sent_; }

 private:
  holoscan::Parameter<int> message_count_;
  int sent_ = 0;
  int compute_count_ = 0;
};

/// Receives integer values from a pub/sub topic and counts them.
class RxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RxOp)
  RxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<int>("in").topic("ping"); }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto value = op_input.receive<int>("in");
    if (value) {
      int count = ++received_;
      HOLOSCAN_LOG_INFO("[RxOp] compute #{}: received value {}, total received: {}",
                        compute_count_,
                        value.value(),
                        count);
    } else {
      HOLOSCAN_LOG_INFO("[RxOp] compute #{}: receive returned empty, total received: {}",
                        compute_count_,
                        received_.load());
    }
    ++compute_count_;
  }

  int received() const { return received_; }

 private:
  std::atomic<int> received_{0};
  int compute_count_ = 0;
};

/// Receives integer values from a directly topic-bound kAnySize input.
class RxAnySizeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RxAnySizeOp)
  RxAnySizeOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::vector<int>>("receivers", holoscan::IOSpec::kAnySize);
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto values = op_input.receive<std::vector<int>>("receivers");
    if (values) {
      const auto batch_size = static_cast<int>(values->size());
      total_received_.fetch_add(batch_size);
      if (values->size() != 1) {
        saw_non_singleton_batch_.store(true);
      }
      HOLOSCAN_LOG_INFO("[RxAnySizeOp] compute #{}: received batch size {}, total received: {}",
                        compute_count_,
                        values->size(),
                        total_received_.load());
    } else {
      HOLOSCAN_LOG_INFO("[RxAnySizeOp] compute #{}: receive returned empty, total received: {}",
                        compute_count_,
                        total_received_.load());
    }
    ++compute_count_;
  }

  int received() const { return total_received_.load(); }
  bool saw_non_singleton_batch() const { return saw_non_singleton_batch_.load(); }

 private:
  std::atomic<int> total_received_{0};
  std::atomic<bool> saw_non_singleton_batch_{false};
  int compute_count_ = 0;
};

//==============================================================================
// Application
//==============================================================================

struct TestParam {
  std::string scheduler;
};

std::string ParamName(const ::testing::TestParamInfo<TestParam>& info) {
  return info.param.scheduler;
}

class PubSubLoopbackApp : public holoscan::Application {
 public:
  explicit PubSubLoopbackApp(const TestParam& param) : param_(param) {}

  /// Provide the in-memory backend instead of the default (no-op) backend.
  /// This is the same extensibility hook that a third-party backend would use.
  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    return make_network_context<holoscan::InMemoryPubSubNetworkContext>("pubsub_ctx");
  }

  void compose() override {
    using namespace holoscan;

    // Short deadlock timeout: in-memory delivery is synchronous, no async
    // discovery latency.
    const int64_t deadlock_timeout_ms = 1000;

    if (param_.scheduler == "greedy") {
      scheduler(make_scheduler<GreedyScheduler>(
          "scheduler", Arg("stop_on_deadlock_timeout", deadlock_timeout_ms)));
    } else {
      scheduler(make_scheduler<EventBasedScheduler>(
          "scheduler",
          Arg("worker_thread_number", static_cast<int64_t>(2)),
          Arg("stop_on_deadlock_timeout", deadlock_timeout_ms),
          Arg("enable_queue_stealing", false),
          Arg("enable_worker_postcheck_fastpath", false),
          Arg("internal_event_shard_count", static_cast<int64_t>(1)),
          Arg("dispatcher_internal_pop_batch_size", static_cast<int64_t>(1)),
          Arg("wait_state_shard_count", static_cast<int64_t>(1))));
    }

    // SubscriberAvailableCondition gates the Tx until the Rx endpoint has
    // registered itself in the in-memory discovery service.
    auto sub_avail = make_condition<SubscriberAvailableCondition>(
        "sub_avail",
        Arg("transmitter", std::string("out")),
        Arg("min_subscriber_count", static_cast<uint64_t>(1)));

    // PeriodicCondition throttles Tx to prevent it from outrunning Rx.
    // Without this, the EventBasedScheduler can dispatch Tx twice before
    // Rx runs, overflowing PubSubReceiver's capacity-1 staging queue and
    // silently dropping the older message (see GXF_PUBSUB_RECEIVER_SYNC).
    auto periodic = make_condition<PeriodicCondition>("periodic", std::chrono::milliseconds(10));

    tx_ = make_operator<TxOp>("tx",
                              sub_avail,
                              periodic,
                              make_condition<CountCondition>("count", kMessageCount),
                              Arg("message_count", kMessageCount));
    rx_ = make_operator<RxOp>("rx");

    add_operator(tx_);
    add_operator(rx_);
  }

  std::shared_ptr<TxOp> tx() const { return tx_; }
  std::shared_ptr<RxOp> rx() const { return rx_; }

  static constexpr int kMessageCount = 100;

 private:
  TestParam param_;
  std::shared_ptr<TxOp> tx_;
  std::shared_ptr<RxOp> rx_;
};

class PubSubAnySizeLoopbackApp : public holoscan::Application {
 public:
  explicit PubSubAnySizeLoopbackApp(const TestParam& param) : param_(param) {}

  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    return make_network_context<holoscan::InMemoryPubSubNetworkContext>("pubsub_ctx");
  }

  void compose() override {
    using namespace holoscan;

    const int64_t deadlock_timeout_ms = 1000;

    if (param_.scheduler == "greedy") {
      scheduler(make_scheduler<GreedyScheduler>(
          "scheduler", Arg("stop_on_deadlock_timeout", deadlock_timeout_ms)));
    } else {
      scheduler(make_scheduler<EventBasedScheduler>(
          "scheduler",
          Arg("worker_thread_number", static_cast<int64_t>(2)),
          Arg("stop_on_deadlock_timeout", deadlock_timeout_ms),
          Arg("enable_queue_stealing", false),
          Arg("enable_worker_postcheck_fastpath", false),
          Arg("internal_event_shard_count", static_cast<int64_t>(1)),
          Arg("dispatcher_internal_pop_batch_size", static_cast<int64_t>(1)),
          Arg("wait_state_shard_count", static_cast<int64_t>(1))));
    }

    auto sub_avail = make_condition<SubscriberAvailableCondition>(
        "sub_avail",
        Arg("transmitter", std::string("out")),
        Arg("min_subscriber_count", static_cast<uint64_t>(1)));
    auto periodic = make_condition<PeriodicCondition>("periodic", std::chrono::milliseconds(10));

    tx_ = make_operator<TxOp>("tx",
                              sub_avail,
                              periodic,
                              make_condition<CountCondition>("count", kMessageCount),
                              Arg("message_count", kMessageCount));
    rx_ = make_operator<RxAnySizeOp>("rx");
    rx_->bind_input_topic("receivers", "ping");

    add_operator(tx_);
    add_operator(rx_);
  }

  std::shared_ptr<TxOp> tx() const { return tx_; }
  std::shared_ptr<RxAnySizeOp> rx() const { return rx_; }

  static constexpr int kMessageCount = 100;

 private:
  TestParam param_;
  std::shared_ptr<TxOp> tx_;
  std::shared_ptr<RxAnySizeOp> rx_;
};

class PubSubAnySizeMixedApp : public holoscan::Application {
 public:
  std::shared_ptr<holoscan::NetworkContext> create_pubsub_network_context() override {
    return make_network_context<holoscan::InMemoryPubSubNetworkContext>("pubsub_ctx");
  }

  void compose() override {
    using namespace holoscan;

    scheduler(make_scheduler<GreedyScheduler>(
        "scheduler", Arg("stop_on_deadlock_timeout", static_cast<int64_t>(1000))));

    tx_ = make_operator<TxOp>(
        "tx", make_condition<CountCondition>("count", 1), Arg("message_count", 1));
    rx_ = make_operator<RxAnySizeOp>("rx");
    rx_->bind_input_topic("receivers", "ping");

    add_flow(tx_, rx_, {{"out", "receivers"}});
  }

 private:
  std::shared_ptr<TxOp> tx_;
  std::shared_ptr<RxAnySizeOp> rx_;
};

//==============================================================================
// Parameterized test
//==============================================================================

class PubSubLoopbackTest : public ::testing::TestWithParam<TestParam> {};

TEST_P(PubSubLoopbackTest, AllMessagesDelivered) {
  const auto& param = GetParam();
  auto app = holoscan::make_application<PubSubLoopbackApp>(param);
  app->run();

  auto tx = app->tx();
  auto rx = app->rx();
  ASSERT_NE(tx, nullptr);
  ASSERT_NE(rx, nullptr);

  HOLOSCAN_LOG_INFO("=== FINAL: tx sent {}, rx received {} (expected {}) ===",
                    tx->sent(),
                    rx->received(),
                    PubSubLoopbackApp::kMessageCount);

  // In-memory kPassthrough mode: every message must arrive.
  EXPECT_EQ(rx->received(), PubSubLoopbackApp::kMessageCount)
      << "Expected all " << PubSubLoopbackApp::kMessageCount
      << " messages via in-memory pub/sub (scheduler: " << param.scheduler << ")";
}

TEST_P(PubSubLoopbackTest, TopicBoundAnySizeInputReceivesAllMessages) {
  const auto& param = GetParam();
  auto app = holoscan::make_application<PubSubAnySizeLoopbackApp>(param);
  app->run();

  auto tx = app->tx();
  auto rx = app->rx();
  ASSERT_NE(tx, nullptr);
  ASSERT_NE(rx, nullptr);

  EXPECT_EQ(rx->received(), PubSubAnySizeLoopbackApp::kMessageCount)
      << "Expected all " << PubSubAnySizeLoopbackApp::kMessageCount
      << " messages via directly topic-bound kAnySize pub/sub input (scheduler: " << param.scheduler
      << ")";
  EXPECT_FALSE(rx->saw_non_singleton_batch());
}

TEST_P(PubSubLoopbackTest, TopicBoundAnySizeMixedWithIndexedPortsIsRejected) {
  auto app = holoscan::make_application<PubSubAnySizeMixedApp>();
  EXPECT_THROW(app->run(), std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(Schedulers, PubSubLoopbackTest,
                         ::testing::Values(TestParam{"greedy"}, TestParam{"event_based"}),
                         ParamName);

}  // namespace
