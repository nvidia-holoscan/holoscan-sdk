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

#include <chrono>
#include <memory>
#include <random>
#include <thread>

#include <holoscan/holoscan.hpp>

// 0. State definition.
class SensorState {
 public:
  SensorState() = default;

  [[nodiscard]] float get_sensor_value() const { return sensor_value; }
  void set_sensor_value(float value) { sensor_value = value; }

 private:
  float sensor_value = 0.0F;
};

// 1. SensorOp: Simulates a sensor reading and stores it in a member variable.
class SensorOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SensorOp)
  SensorOp() = default;

  // Member variable for the sensor reading.
  [[nodiscard]] float get_sensor_value() const { return sensor_value_; }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Simulate a sensor reading between 0 and 100.
    sensor_value_ = distribution_(generator_);
    auto state = metadata()->get<std::shared_ptr<SensorState>>("state");
    state->set_sensor_value(sensor_value_);

    HOLOSCAN_LOG_INFO("{} - Sensor reading: {}", name(), sensor_value_);
    // Sleep to simulate sensor frequency.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

 private:
  std::default_random_engine generator_{std::random_device{}()};  // NOLINT(whitespace/braces)
  std::uniform_real_distribution<float> distribution_{0.0F, 100.0F};
  float sensor_value_ = 0.0F;
};

// 2. PreprocessOp: Simulates a processing delay.
class PreprocessOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PreprocessOp)
  PreprocessOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Get the sensor value from the metadata.
    auto state = metadata()->get<std::shared_ptr<SensorState>>("state");
    auto sensor_value = state->get_sensor_value();
    HOLOSCAN_LOG_INFO("{} - Preprocessing... Sensor value: {}", name(), sensor_value);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
};

// 3. AlertOp: Increments an alert counter each time it is triggered.
class AlertOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AlertOp)
  AlertOp() = default;

  int get_alert_count() const { return alert_count_; }
  void set_alert_count(int alert_count) { alert_count_ = alert_count; }

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Get the sensor value from the metadata.
    auto state = metadata()->get<std::shared_ptr<SensorState>>("state");
    auto sensor_value = state->get_sensor_value();
    alert_count_++;
    HOLOSCAN_LOG_WARN(
        "{} - ALERT triggered! Sensor value: {}, Count: {}", name(), sensor_value, alert_count_);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
  }

 private:
  int alert_count_ = 0;
};

// 4. RecoveryOp: Simulates recovery procedures.
class RecoveryOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(RecoveryOp)
  RecoveryOp() = default;

  void compute([[maybe_unused]] holoscan::InputContext& op_input,
               [[maybe_unused]] holoscan::OutputContext& op_output,
               [[maybe_unused]] holoscan::ExecutionContext& context) override {
    // Get the sensor value from the metadata.
    auto state = metadata()->get<std::shared_ptr<SensorState>>("state");
    auto sensor_value = state->get_sensor_value();
    HOLOSCAN_LOG_INFO("{} - Executing recovery procedures. Sensor value: {}", name(), sensor_value);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
};

class SensorMonitoringApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create the operators.
    auto sensor = make_operator<SensorOp>("sensor_op");
    auto preprocess = make_operator<PreprocessOp>("preprocess_op");
    auto alert = make_operator<AlertOp>("alert_op");
    auto recovery = make_operator<RecoveryOp>("recovery_op");

    // Define the state-based monitoring workflow
    //
    // Node Graph:
    //
    //                (cycle)
    //                -------
    //                \     /
    // <|start|> -> SensorOp -> PreprocessOp -> (SensorOp)
    //               (normal)        |
    //                               |(abnormal)
    //                               -> AlertOp --------------------> (SensorOp)
    //                                      |     (alert_count < 3)
    //                                      |
    //                                      --------> RecoveryOp ------> (SensorOp)
    //                                            (alert_count >= 3)

    // The static flow: start_op() triggers sensor_op.
    add_flow(start_op(), sensor);
    add_flow(sensor, preprocess);
    add_flow(sensor, alert);
    add_flow(preprocess, sensor);
    add_flow(alert, sensor);
    add_flow(alert, recovery);
    add_flow(recovery, sensor);

    set_dynamic_flows(start_op(), [](const std::shared_ptr<Operator>& op) {
      // Set global state to the metadata.
      op->metadata()->set<std::shared_ptr<SensorState>>("state", std::make_shared<SensorState>());
      // Route to all connected flows.
      op->add_dynamic_flow(op->next_flows());
    });

    // As sensor_op operates in a loop, the following metadata policy is internally configured
    // to permit metadata updates during the application's execution:
    //   sensor_op->metadata_policy(holoscan::MetadataPolicy::kUpdate);

    // Set dynamic flows for SensorOp.
    // Depending on the sensor_value (stored in its member variable),
    // SensorOp will add a dynamic flow to either PreprocessOp (normal) or AlertOp (abnormal).
    const int kMaxCount = 10;
    set_dynamic_flows(sensor, [preprocess, alert, kMaxCount](const std::shared_ptr<Operator>& op) {
      static int count = 0;

      count++;
      if (count > kMaxCount) {
        HOLOSCAN_LOG_INFO(
            "{} - count is larger than {}. Finishing the dynamic flow.", op->name(), kMaxCount);
        return;
      }

      auto sensor = std::static_pointer_cast<SensorOp>(op);
      if (sensor->get_sensor_value() < 20. || sensor->get_sensor_value() > 80.0F) {
        HOLOSCAN_LOG_INFO(
            "{} - sensor value is abnormal ({:.2f} is not in range [20, 80]). Adding dynamic "
            "flow to AlertOp.",
            op->name(),
            sensor->get_sensor_value());

        sensor->add_dynamic_flow(alert);
      } else {
        sensor->add_dynamic_flow(preprocess);
      }
    });

    // Set dynamic flows for AlertOp:
    // If the alert_count is less than 3, return to SensorOp.
    // Otherwise, trigger RecoveryOp.
    set_dynamic_flows(alert, [recovery, sensor](const std::shared_ptr<Operator>& op) {
      auto alert = std::static_pointer_cast<AlertOp>(op);
      if (alert->get_alert_count() >= 3) {
        HOLOSCAN_LOG_INFO("{} - Alert count is larger than 3. Adding dynamic flow to RecoveryOp.",
                          op->name());
        alert->set_alert_count(0);
        alert->add_dynamic_flow(recovery);
      } else {
        HOLOSCAN_LOG_INFO("{} - Alert count is less than 3. Adding dynamic flow to SensorOp.",
                          op->name());
        alert->add_dynamic_flow(sensor);
      }
    });
  }
};

int main() {
  auto app = holoscan::make_application<SensorMonitoringApp>();
  app->run();
  return 0;
}

// Expected output:
// (The output is non-deterministic due to random sensor values, so the exact values and order
// may vary)
//
// sensor_op - Sensor reading: 26.981735
// preprocess_op - Preprocessing... Sensor value: 26.981735
// sensor_op - Sensor reading: 82.03595
// sensor_op - sensor value is abnormal (82.04 is not in range [20, 80]).
// Adding dynamic flow to AlertOp.
// alert_op - ALERT triggered! Sensor value: 82.03595, Count: 1
// alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
// sensor_op - Sensor reading: 78.258675
// preprocess_op - Preprocessing... Sensor value: 78.258675
// sensor_op - Sensor reading: 93.5977
// sensor_op - sensor value is abnormal (93.60 is not in range [20, 80]).
// Adding dynamic flow to AlertOp.
// alert_op - ALERT triggered! Sensor value: 93.5977, Count: 2
// alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
// sensor_op - Sensor reading: 96.47388
// sensor_op - sensor value is abnormal (96.47 is not in range [20, 80]).
// Adding dynamic flow to AlertOp.
// alert_op - ALERT triggered! Sensor value: 96.47388, Count: 3
// alert_op - Alert count is larger than 3. Adding dynamic flow to RecoveryOp.
// recovery_op - Executing recovery procedures. Sensor value: 96.47388
// sensor_op - Sensor reading: 36.48345
// preprocess_op - Preprocessing... Sensor value: 36.48345
// sensor_op - Sensor reading: 77.37341
// preprocess_op - Preprocessing... Sensor value: 77.37341
// sensor_op - Sensor reading: 14.930141
// sensor_op - sensor value is abnormal (14.93 is not in range [20, 80]).
// Adding dynamic flow to AlertOp.
// alert_op - ALERT triggered! Sensor value: 14.930141, Count: 1
// alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
// sensor_op - Sensor reading: 30.892742
// preprocess_op - Preprocessing... Sensor value: 30.892742
// sensor_op - Sensor reading: 14.287746
// sensor_op - sensor value is abnormal (14.29 is not in range [20, 80]).
// Adding dynamic flow to AlertOp.
// alert_op - ALERT triggered! Sensor value: 14.287746, Count: 2
// alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
// sensor_op - Sensor reading: 34.149467
// sensor_op - count is larger than 10. Finishing the dynamic flow.
