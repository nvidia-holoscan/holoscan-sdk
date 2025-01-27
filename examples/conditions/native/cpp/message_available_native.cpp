/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/ping_rx/ping_rx.hpp"
#include "holoscan/operators/ping_tx/ping_tx.hpp"

namespace holoscan::conditions {

class NativeMessageAvailableCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(NativeMessageAvailableCondition)

  NativeMessageAvailableCondition() = default;

  void initialize() override {
    // call parent initialize or parameters will not be registered
    Condition::initialize();
  };

  void setup(ComponentSpec& spec) override {
    spec.param(receiver_,
               "receiver",
               "Receiver",
               "The scheduling term permits execution if this channel has at least a given "
               "number of messages available.");
    spec.param(min_size_,
               "min_size",
               "Minimum size",
               "The condition permits execution if the given receiver has at least the given "
               "number of messages available",
               static_cast<uint64_t>(1));
  }

  void check(int64_t timestamp, SchedulingStatusType* type,
             int64_t* target_timestamp) const override {
    if (type == nullptr) {
      throw std::runtime_error(fmt::format("Condition '{}' received nullptr for type", name()));
    }
    if (target_timestamp == nullptr) {
      throw std::runtime_error(
          fmt::format("Condition '{}' received nullptr for target_timestamp", name()));
    }
    *type = current_state_;
    *target_timestamp = last_state_change_;
  };

  void on_execute(int64_t timestamp) override { update_state(timestamp); };

  void update_state(int64_t timestamp) override {
    const bool is_ready = check_min_size();
    if (is_ready && current_state_ != SchedulingStatusType::kReady) {
      current_state_ = SchedulingStatusType::kReady;
      last_state_change_ = timestamp;
    }

    if (!is_ready && current_state_ != SchedulingStatusType::kWait) {
      current_state_ = SchedulingStatusType::kWait;
      last_state_change_ = timestamp;
    }
  };

 private:
  bool check_min_size() {
    auto recv = receiver_.get();
    return recv->back_size() + recv->size() >= min_size_.get();
  }

  Parameter<std::shared_ptr<holoscan::Receiver>> receiver_;
  Parameter<uint64_t> min_size_;

  SchedulingStatusType current_state_ =
      SchedulingStatusType::kWait;  // The current state of the scheduling term
  int64_t last_state_change_ = 0;   // timestamp when the state changed the last time
};

}  // namespace holoscan::conditions

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto tx =
        make_operator<ops::PingTxOp>("tx", make_condition<CountCondition>("count-condition", 10));

    auto message_cond = make_condition<conditions::NativeMessageAvailableCondition>(
        "in_native_message_available",
        Arg("receiver", "in"),
        Arg("min_size", static_cast<uint64_t>(1)));

    auto rx = make_operator<ops::PingRxOp>("rx", message_cond);

    add_flow(tx, rx);
  }
};

int main([[maybe_unused]] int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  app->run();

  return 0;
}
