"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import random
import time
from dataclasses import dataclass

from holoscan.core import Application, Operator


# 0. State definition
@dataclass
class SensorState:
    """Class to store sensor state."""

    sensor_value: float = 0.0


# 1. SensorOp: Simulates a sensor reading and stores it in a member variable
class SensorOp(Operator):
    """Simulates a sensor reading and stores it in a member variable."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_value = 0.0

    def compute(self, op_input, op_output, context):
        # Simulate a sensor reading between 0 and 100
        self.sensor_value = random.uniform(0.0, 100.0)
        state = self.metadata.get("state")
        state.sensor_value = self.sensor_value

        print(f"{self.name} - Sensor reading: {self.sensor_value}")
        # Sleep to simulate sensor frequency
        time.sleep(0.05)  # 50ms


## Alternative way to define the operator
# from holoscan.decorator import create_op


# @create_op(op_param="self")
# def sensor_op(self):
#     self.sensor_value = random.uniform(0.0, 100.0)
#     state = self.metadata.get("state")
#     state.sensor_value = self.sensor_value
#     print(f"{self.name} - Sensor reading: {self.sensor_value}")
#     time.sleep(0.05)  # 50ms


# 2. PreprocessOp: Simulates a processing delay
class PreprocessOp(Operator):
    """Simulates a processing delay."""

    def compute(self, op_input, op_output, context):
        # Get the sensor value from the metadata
        state = self.metadata.get("state")
        sensor_value = state.sensor_value
        print(f"{self.name} - Preprocessing... Sensor value: {sensor_value}")
        time.sleep(0.05)  # 50ms


## Alternative way to define the operator
# @create_op(op_param="self")
# def preprocess_op(self):
#     state = self.metadata.get("state")
#     sensor_value = state.sensor_value
#     print(f"{self.name} - Preprocessing... Sensor value: {sensor_value}")
#     time.sleep(0.05)  # 50ms


# 3. AlertOp: Increments an alert counter each time it is triggered
class AlertOp(Operator):
    """Increments an alert counter each time it is triggered."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alert_count = 0

    def compute(self, op_input, op_output, context):
        # Get the sensor value from the metadata
        state = self.metadata.get("state")
        sensor_value = state.sensor_value
        self.alert_count += 1
        print(
            f"{self.name} - ALERT triggered! Sensor value: {sensor_value}, "
            f"Count: {self.alert_count}"
        )
        time.sleep(0.03)  # 30ms


## Alternative way to define the operator
# @create_op(op_param="self")
# def alert_op(self):
#     if not hasattr(self, "alert_count"):
#         self.alert_count = 0
#     state = self.metadata.get("state")
#     sensor_value = state.sensor_value
#     self.alert_count += 1
#     print(f"{self.name} - ALERT triggered! Sensor value: {sensor_value}, Count: {self.alert_count}")
#     time.sleep(0.03)  # 30ms


# 4. RecoveryOp: Simulates recovery procedures
class RecoveryOp(Operator):
    """Simulates recovery procedures."""

    def compute(self, op_input, op_output, context):
        # Get the sensor value from the metadata
        state = self.metadata.get("state")
        sensor_value = state.sensor_value
        print(f"{self.name} - Executing recovery procedures. Sensor value: {sensor_value}")
        time.sleep(0.1)  # 100ms


## Alternative way to define the operator
# @create_op(op_param="self")
# def recovery_op(self):
#     state = self.metadata.get("state")
#     sensor_value = state.sensor_value
#     print(f"{self.name} - Executing recovery procedures. Sensor value: {sensor_value}")
#     time.sleep(0.1)  # 100ms


class SensorMonitoringApp(Application):
    def compose(self):
        # Create the operators
        sensor = SensorOp(self, name="sensor_op")
        preprocess = PreprocessOp(self, name="preprocess_op")
        alert = AlertOp(self, name="alert_op")
        recovery = RecoveryOp(self, name="recovery_op")

        ## If decorator is used, the operator is defined as follows:
        # sensor = sensor_op(self, name="sensor_op")
        # preprocess = preprocess_op(self, name="preprocess_op")
        # alert = alert_op(self, name="alert_op")
        # recovery = recovery_op(self, name="recovery_op")

        # Define the state-based monitoring workflow
        #
        # Node Graph:
        #
        #                (cycle)
        #                -------
        #                \     /
        # <|start|> -> SensorOp -> PreprocessOp -> (SensorOp)
        #               (normal)        |
        #                               |(abnormal)
        #                               -> AlertOp --------------------> (SensorOp)
        #                                      |     (alert_count < 3)
        #                                      |
        #                                      --------> RecoveryOp ------> (SensorOp)
        #                                            (alert_count >= 3)

        # The static flow: start_op() triggers sensor_op
        self.add_flow(self.start_op(), sensor)
        self.add_flow(sensor, preprocess)
        self.add_flow(sensor, alert)
        self.add_flow(preprocess, sensor)
        self.add_flow(alert, sensor)
        self.add_flow(alert, recovery)
        self.add_flow(recovery, sensor)

        def start_op_callback(op):
            # Set global state to the metadata
            op.metadata.set("state", SensorState())
            # Route to all connected flows
            op.add_dynamic_flow(op.next_flows)

        # Set dynamic flows for start_op
        self.set_dynamic_flows(self.start_op(), start_op_callback)

        # As sensor_op operates in a loop, the following metadata policy is internally configured
        # to permit metadata updates during the application's execution:
        #   sensor_op.metadata_policy(MetadataPolicy.UPDATE)

        # Set dynamic flows for SensorOp
        # Depending on the sensor_value (stored in its member variable),
        # SensorOp will add a dynamic flow to either PreprocessOp (normal) or AlertOp (abnormal)
        def sensor_op_callback(op):
            if not hasattr(sensor_op_callback, "count"):
                sensor_op_callback.count = 0

            sensor_op_callback.count += 1
            if sensor_op_callback.count > 10:
                print(f"{op.name} - count is larger than 10. Finishing the dynamic flow.")
                return

            if op.sensor_value < 20.0 or op.sensor_value > 80.0:
                print(
                    f"{op.name} - sensor value is abnormal ({op.sensor_value:.2f} is not in range "
                    "[20, 80]). Adding dynamic flow to AlertOp."
                )
                op.alert_count = 0
                op.add_dynamic_flow(alert)
            else:
                op.add_dynamic_flow(preprocess)

        self.set_dynamic_flows(sensor, sensor_op_callback)

        # Set dynamic flows for AlertOp:
        # If the alert_count is less than 3, return to SensorOp.
        # Otherwise, trigger RecoveryOp.
        def alert_op_callback(op):
            if op.alert_count >= 3:
                print(
                    f"{op.name} - Alert count is larger than 3. Adding dynamic flow to RecoveryOp."
                )
                op.add_dynamic_flow(recovery)
            else:
                print(f"{op.name} - Alert count is less than 3. Adding dynamic flow to SensorOp.")
                op.add_dynamic_flow(sensor)

        self.set_dynamic_flows(alert, alert_op_callback)


def main():
    app = SensorMonitoringApp()
    app.run()


if __name__ == "__main__":
    main()

# NOLINTBEGIN
# ruff: noqa: E501
#
# Expected output:
# (The output is non-deterministic due to random sensor values, so the exact values and order may vary)
#
# sensor_op - Sensor reading: 26.981735
# preprocess_op - Preprocessing... Sensor value: 26.981735
# sensor_op - Sensor reading: 82.03595
# sensor_op - sensor value is abnormal (82.04 is not in range [20, 80]). Adding dynamic flow to AlertOp.
# alert_op - ALERT triggered! Sensor value: 82.03595, Count: 1
# alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
# sensor_op - Sensor reading: 78.258675
# preprocess_op - Preprocessing... Sensor value: 78.258675
# sensor_op - Sensor reading: 93.5977
# sensor_op - sensor value is abnormal (93.60 is not in range [20, 80]). Adding dynamic flow to AlertOp.
# alert_op - ALERT triggered! Sensor value: 93.5977, Count: 2
# alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
# sensor_op - Sensor reading: 96.47388
# sensor_op - sensor value is abnormal (96.47 is not in range [20, 80]). Adding dynamic flow to AlertOp.
# alert_op - ALERT triggered! Sensor value: 96.47388, Count: 3
# alert_op - Alert count is larger than 3. Adding dynamic flow to RecoveryOp.
# recovery_op - Executing recovery procedures. Sensor value: 96.47388
# sensor_op - Sensor reading: 36.48345
# preprocess_op - Preprocessing... Sensor value: 36.48345
# sensor_op - Sensor reading: 77.37341
# preprocess_op - Preprocessing... Sensor value: 77.37341
# sensor_op - Sensor reading: 14.930141
# sensor_op - sensor value is abnormal (14.93 is not in range [20, 80]). Adding dynamic flow to AlertOp.
# alert_op - ALERT triggered! Sensor value: 14.930141, Count: 1
# alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
# sensor_op - Sensor reading: 30.892742
# preprocess_op - Preprocessing... Sensor value: 30.892742
# sensor_op - Sensor reading: 14.287746
# sensor_op - sensor value is abnormal (14.29 is not in range [20, 80]). Adding dynamic flow to AlertOp.
# alert_op - ALERT triggered! Sensor value: 14.287746, Count: 2
# alert_op - Alert count is less than 3. Adding dynamic flow to SensorOp.
# sensor_op - Sensor reading: 34.149467
# sensor_op - count is larger than 10. Finishing the dynamic flow.
#
# ruff: qa
# NOLINTEND
