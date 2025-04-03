/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/ping_rx/ping_rx.hpp>
#include <holoscan/operators/ping_tx/ping_tx.hpp>

namespace holoscan {

namespace {

class VerbosePeriodicCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(VerbosePeriodicCondition)

  VerbosePeriodicCondition() = default;

  void initialize() override {
    // call parent initialize or parameters will not be registered
    Condition::initialize();
    recess_period_ns_ = recess_period_.get();
  };

  void setup(ComponentSpec& spec) override {
    spec.param(recess_period_,
               "recess_period",
               "Recess Period",
               "Recession period in nanoseconds",
               static_cast<int64_t>(0));
  }

  void check(int64_t timestamp, SchedulingStatusType* status_type,
             int64_t* target_timestamp) const override {
    HOLOSCAN_LOG_INFO("In check for condition {}", name());
    if (status_type == nullptr) {
      throw std::runtime_error(
          fmt::format("Condition '{}' received nullptr for status_type", name()));
    }
    if (target_timestamp == nullptr) {
      throw std::runtime_error(
          fmt::format("Condition '{}' received nullptr for target_timestamp", name()));
    }
    if (!next_target_.has_value()) {
      *status_type = SchedulingStatusType::kReady;
      *target_timestamp = timestamp;
      return;
    }
    *target_timestamp = next_target_.value();
    *status_type = timestamp > *target_timestamp ? SchedulingStatusType::kReady
                                                 : SchedulingStatusType::kWaitTime;
  };

  void on_execute(int64_t timestamp) override {
    HOLOSCAN_LOG_INFO("In on_execute for condition {}", name());
    if (next_target_.has_value()) {
      next_target_ = next_target_.value() + recess_period_ns_;
    } else {
      next_target_ = timestamp + recess_period_ns_;
    }
  };

  void update_state([[maybe_unused]] int64_t timestamp) override {
    // no-op for this condition
  };

 private:
  Parameter<int64_t> recess_period_;

  int64_t recess_period_ns_ = 0;
  std::optional<int64_t> next_target_ = std::nullopt;
};

class VerboseCountCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(VerboseCountCondition)

  VerboseCountCondition() = default;

  void initialize() override {
    // call parent initialize or parameters will not be registered
    Condition::initialize();
    remaining_ = count_.get();
    current_state_ = SchedulingStatusType::kReady;
    last_run_timestamp_ = 0;
  };

  void setup(ComponentSpec& spec) override {
    spec.param(count_,
               "count",
               "Count",
               "The total number of time this term will permit execution.",
               static_cast<int64_t>(0));
  }

  void check(int64_t timestamp, SchedulingStatusType* status_type,
             int64_t* target_timestamp) const override {
    HOLOSCAN_LOG_INFO("In check for condition {}, setting status_type to {} and timestamp to {}",
                      name(),
                      static_cast<int>(current_state_),
                      last_run_timestamp_);
    *status_type = current_state_;
    *target_timestamp = last_run_timestamp_;
    return;
  };

  void on_execute(int64_t timestamp) override {
    HOLOSCAN_LOG_INFO("In on_execute for condition {}, remaining = {}", name(), remaining_);
    remaining_--;
    if (remaining_ == 0) { current_state_ = SchedulingStatusType::kNever; }
    last_run_timestamp_ = timestamp;
    return;
  };

  void update_state([[maybe_unused]] int64_t timestamp) override {
    HOLOSCAN_LOG_INFO("In update_state for condition {}, remaining = {}", name(), remaining_);
    if (remaining_ == 0) { current_state_ = SchedulingStatusType::kNever; }
    return;
  };

 private:
  Parameter<int64_t> count_{};

  int64_t remaining_{};  ///< The remaining number of permitted executions.
  SchedulingStatusType current_state_ = SchedulingStatusType::kReady;  ///< current scheduling state
  int64_t last_run_timestamp_{};  ///< timestamp when the entity was last executed
};

class OrCombinerCountConditionsApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    using namespace std::chrono_literals;

    std::vector<std::shared_ptr<Condition>> terms;
    if (use_native_count_conditions_) {
      auto count_condition =
          make_condition<VerboseCountCondition>("tx_count", Arg("count", static_cast<int64_t>(10)));
      auto count_condition2 = make_condition<VerboseCountCondition>(
          "tx_count2", Arg("count", static_cast<int64_t>(20)));
      terms.push_back(count_condition);
      terms.push_back(count_condition2);
    } else {
      auto count_condition =
          make_condition<CountCondition>("tx_count", Arg("count", static_cast<int64_t>(10)));
      auto count_condition2 =
          make_condition<CountCondition>("tx_count2", Arg("count", static_cast<int64_t>(20)));
      terms.push_back(count_condition);
      terms.push_back(count_condition2);
    }

    // create the resource needed to switch those two to OR combination
    auto or_combiner = make_resource<OrConditionCombiner>("or_combiner", Arg{"terms", terms});

    auto periodic_condition = make_condition<VerbosePeriodicCondition>(
        "periodic-condition", Arg("recess_period", static_cast<int64_t>(100'000'000)));

    auto tx = make_operator<ops::PingTxOp>("tx", or_combiner, periodic_condition);
    auto rx = make_operator<ops::PingRxOp>("rx");
    add_flow(tx, rx);
  }
  void use_native_count_conditions(bool use_native_count_conditions) {
    use_native_count_conditions_ = use_native_count_conditions;
  }

 private:
  Parameter<bool> use_native_count_conditions_{false};
};

}  // namespace

}  // namespace holoscan

class OrCombinerAppTest : public testing::TestWithParam<bool> {};

TEST_P(OrCombinerAppTest, TestHoloscanCountConditionCombine) {
  using namespace holoscan;

  bool use_native_count_conditions = GetParam();
  auto app = make_application<OrCombinerCountConditionsApp>();
  app->use_native_count_conditions(use_native_count_conditions);

  // capture output so that we can check that the expected value is present
  testing::internal::CaptureStderr();

  try {
    app->run();
  } catch (const std::exception& e) { HOLOSCAN_LOG_ERROR("Exception: {}", e.what()); }

  std::string log_output = testing::internal::GetCapturedStderr();
  EXPECT_TRUE(log_output.find("Rx message value: 10") != std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
  // will have stopped after count of 10 was reached due to the first count condition
  EXPECT_TRUE(log_output.find("Rx message value: 11") == std::string::npos)
      << "=== LOG ===\n"
      << log_output << "\n===========\n";
}

// Run the test with both false and true values for use_native_count_conditions
INSTANTIATE_TEST_SUITE_P(OrCombinerAppTests, OrCombinerAppTest, testing::Values(false, true),
                         [](const testing::TestParamInfo<bool>& info) {
                           return info.param ? "NativeCountCondition" : "HoloscanCountCondition";
                         });
