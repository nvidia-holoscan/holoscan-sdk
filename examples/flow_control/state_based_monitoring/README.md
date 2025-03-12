# State-Based Monitoring Example

This example demonstrates how to implement a state-based monitoring system using Holoscan SDK's dynamic flow control capabilities. The application simulates a sensor monitoring system with alert and recovery procedures.

## Overview

The application consists of four operators that work together to monitor sensor values and respond to abnormal readings:

1. `SensorOp`: Simulates a sensor that generates random values
2. `PreprocessOp`: Processes normal sensor readings
3. `AlertOp`: Handles abnormal readings and tracks alert counts
4. `RecoveryOp`: Executes recovery procedures when multiple alerts occur

## Flow Control

The application uses dynamic flows to route data based on sensor values and alert counts:

```
               (cycle)
               -------
               \     /
<|start|> -> SensorOp -> PreprocessOp -> (SensorOp)
              (normal)        |
                              |(abnormal)
                              -> AlertOp --------------------> (SensorOp)
                                     |     (alert_count < 3)
                                     |
                                     --------> RecoveryOp ------> (SensorOp)
                                           (alert_count >= 3)
```

- If sensor value is normal (20-80): Route to `PreprocessOp`
- If sensor value is abnormal (<20 or >80): Route to `AlertOp`
- If alert count reaches 3: Route to `RecoveryOp`
- After recovery: Return to normal monitoring

## State Management

The application uses a shared state object (`SensorState`) to track sensor values across operators. The state is stored in operator metadata and can be accessed by all operators in the workflow.

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## Building and Running

### C++

```bash
cmake -B build -S .
cmake --build build
./build/examples/flow_control/state_based_monitoring/cpp/state_based_monitoring
```

### Python

```bash
python3 examples/flow_control/state_based_monitoring/python/state_based_monitoring.py
```

Note: The Python example includes commented code that demonstrates an alternative way to define operators using the `@create_op` decorator pattern. You can uncomment these sections to see how the same functionality can be implemented using decorators instead of classes.

## Expected Output

The output will vary due to random sensor values, but will follow this pattern:

```
sensor_op - Sensor reading: 26.981735
preprocess_op - Preprocessing... Sensor value: 26.981735
sensor_op - Sensor reading: 82.03595
sensor_op - sensor value is abnormal (82.04 is not in range [20, 80]). Adding dynamic flow to AlertOp.
alert_op - ALERT triggered! Sensor value: 82.03595, Count: 1
...
```

## Implementation Notes

- The example demonstrates both class-based operators and function-based operators using the `@create_op` decorator (commented out in the code)
- Uses metadata to share state between operators
- Implements dynamic flow control based on both sensor values and alert counts
- Simulates processing delays using sleep intervals
- Runs for 10 iterations before terminating
