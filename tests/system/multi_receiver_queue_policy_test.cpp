/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Reproducer tests for queue policy setting for kAnySize ports (issue 6057656):
 *   IOSpec queue policy is NOT applied to ANY_SIZE (multi-receiver) ports.
 *
 *   MultiReceiverQueuePolicy/PopDoesNotWarn  (RED before fix, GREEN after)
 *     -- fast sender saturates a receiver with policy=kPop; no "Push failed" warning expected.
 *
 *   MultiReceiverQueuePolicy/DefaultFaultDoesWarn  (always GREEN -- regression guard)
 *     -- same topology without setting policy; "Push failed" warning IS expected.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include "holoscan/holoscan.hpp"

namespace {

// ─── Operators ────────────────────────────────────────────────────────────────

/**
 * FastTxOp – emits integers as fast as the scheduler allows.
 * ConditionType::kNone on the output disables DownstreamAffordableCondition so that
 * messages are always pushed regardless of the downstream queue state.
 */
class FastTxOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FastTxOp)

  FastTxOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.output<int>("out").condition(holoscan::ConditionType::kNone);
  }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    op_output.emit(index_++, "out");
  }

 private:
  int index_ = 0;
};

/**
 * MultiReceiverQueuePolicyOp – uses kAnySize on its "receivers" port.
 *
 * The queue policy is set from outside via Operator::queue_policy() after
 * make_operator() returns (setup() is called inside make_operator(), so the
 * policy must be applied to the already-created IOSpec from the app's compose()).
 */
class MultiReceiverQueuePolicyOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MultiReceiverQueuePolicyOp)

  MultiReceiverQueuePolicyOp() = default;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input<std::vector<int>>("receivers", holoscan::IOSpec::kAnySize);
  }

  void compute(holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    auto values = op_input.receive<std::vector<int>>("receivers");
    if (values) {
      HOLOSCAN_LOG_DEBUG("MultiReceiverQueuePolicyOp received {} messages", values->size());
    }
  }
};

// ─── Applications ─────────────────────────────────────────────────────────────

/**
 * Build a minimal app:
 *
 *   tx1 ─┐
 *        ├─→ rx ("receivers", kAnySize, [optional policy])
 *   tx2 ─┘
 *
 * tx1 runs under a PeriodicCondition (slow) and tx2 runs freely (fast), so the
 * "receivers:1" sub-queue attached to tx2 is rapidly saturated.
 */
class MultiReceiverQueuePolicyApp : public holoscan::Application {
 public:
  explicit MultiReceiverQueuePolicyApp(std::optional<holoscan::IOSpec::QueuePolicy> policy,
                                       int count = 30)
      : policy_(policy), count_(count) {}

  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    // tx1 – slow sender (5 Hz)
    constexpr int64_t slow_period_ns = 200'000'000;  // 200 ms → ~5 Hz
    auto tx1 =
        make_operator<FastTxOp>("tx1",
                                make_condition<CountCondition>("count1", count_),
                                make_condition<PeriodicCondition>("periodic1", slow_period_ns));

    // tx2 – fast sender (no periodic constraint, runs as fast as scheduler allows)
    auto tx2 = make_operator<FastTxOp>("tx2", make_condition<CountCondition>("count2", count_));

    auto rx = make_operator<MultiReceiverQueuePolicyOp>("rx");
    // setup() is already called inside make_operator(), so use queue_policy() to set
    // the policy on the already-created IOSpec before add_flow() triggers add_receivers().
    if (policy_.has_value()) {
      rx->queue_policy("receivers", holoscan::IOSpec::IOType::kInput, policy_.value());
    }

    add_flow(tx1, rx, {{"out", "receivers"}});
    add_flow(tx2, rx, {{"out", "receivers"}});
  }

 private:
  std::optional<holoscan::IOSpec::QueuePolicy> policy_;
  int count_;
};

}  // namespace

// ─── Test fixtures ────────────────────────────────────────────────────────────

/**
 * MultiReceiverQueuePolicy/PopDoesNotWarn
 *
 * When policy=kPop is set on the kAnySize IOSpec the fix propagates it to
 * "receivers:0" and "receivers:1".  No "Push failed" warning should appear.
 *
 * EXPECTED: RED before the fix, GREEN after.
 */
TEST(MultiReceiverQueuePolicy, PopDoesNotWarn) {
  auto app = holoscan::make_application<MultiReceiverQueuePolicyApp>(
      holoscan::IOSpec::QueuePolicy::kPop, /*count=*/30);

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "ebs",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(4)),
      holoscan::Arg("stop_on_deadlock", true),
      holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(5000)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(15000))));

  testing::internal::CaptureStderr();
  app->run();
  std::string log_output = testing::internal::GetCapturedStderr();

  // With kPop the receiver silently discards older messages — no "Push failed" warnings.
  EXPECT_EQ(log_output.find("Push failed on receiver 'receivers:"), std::string::npos)
      << "Expected NO 'Push failed on receivers:N' warning when policy=kPop, but found one.\n"
      << "=== STDERR ===\n"
      << log_output << "\n==============\n";
}

/**
 * MultiReceiverQueuePolicy/DefaultFaultDoesWarn
 *
 * When no policy is set the default kFault behaviour applies and "Push failed"
 * warnings SHOULD appear when the fast sender saturates the queue.
 *
 * This is a regression guard — it must always be GREEN.
 */
TEST(MultiReceiverQueuePolicy, DefaultFaultDoesWarn) {
  auto app = holoscan::make_application<MultiReceiverQueuePolicyApp>(std::nullopt, /*count=*/30);

  app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
      "ebs",
      holoscan::Arg("worker_thread_number", static_cast<int64_t>(4)),
      holoscan::Arg("stop_on_deadlock", true),
      holoscan::Arg("stop_on_deadlock_timeout", static_cast<int64_t>(5000)),
      holoscan::Arg("max_duration_ms", static_cast<int64_t>(15000))));

  testing::internal::CaptureStderr();
  app->run();
  std::string log_output = testing::internal::GetCapturedStderr();

  // With default kFault the receiver logs "Push failed on receiver 'receivers:N'" when full.
  EXPECT_NE(log_output.find("Push failed on receiver 'receivers:"), std::string::npos)
      << "Expected at least one 'Push failed on receivers:N' warning"
         " with default policy (kFault).\n"
      << "=== STDERR ===\n"
      << log_output << "\n==============\n";
}
