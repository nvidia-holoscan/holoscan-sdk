# Streaming Monitoring Example with Execution Throttling and Conditional Flow Control

This example demonstrates how to build an application that combines execution throttling and conditional flow control. The application shows how to handle scenarios where different operators process data at different rates and how to conditionally route data based on detected events. Consider the geometry of the application used in this example:

```
                                                              -> report_generation
                                                             |
              (cycle)                                        | (is_detected:true)
              -------         => execution_throttler => detect_event
              \     /         |                              | (is_detected:false)
<|start|> -> gen_signal => process_signal => visualize       |
             (triggered 5 times)                              -> (ignored)
```

The example demonstrates several key flow control patterns:
1. **Execution Throttling**: Using an execution throttler to handle different processing rates between operators
2. **Conditional Flow**: Dynamic routing of data based on event detection
3. **Parallel Processing**: Running visualization and event detection in parallel
4. **Cyclic Flow**: Implementing a cyclic flow pattern with controlled iterations

The key aspects of this application include:
- Configuration of queue policies to prevent push failures
- Use of metadata to share state between operators
- Dynamic flow control based on runtime conditions
- Parallel execution of operators using multi-threaded scheduling

The application demonstrates:
- How to handle operators with different processing rates
- How to implement conditional flows based on detected events
- How to use metadata to share state between operators
- How to configure queue policies for smooth data flow
- How to achieve parallel processing using multi-threaded scheduling

### Flow Control Pattern Selection Guide

This example demonstrates cyclic flow using `start_op()`. Here's when to choose different flow control patterns:

**start_op() + Cyclic Flow**
- Best for: Dynamic routing, feedback loops, runtime-adaptive flows
- Use when: Flow patterns depend on data content or need to change during execution
- Advantages: Flexible, handles complex routing
- Trade-offs: More complex to debug, slightly higher runtime overhead

**Generator (root operator) with condition (CountCondition, PeriodicCondition, etc.)**
- Best for: Fixed iteration counts, simple linear flows
- Use when: Number of iterations is known in advance (or infinite), static flow patterns
- Advantages: Simple to implement, better performance, easier to debug
- Trade-offs: Less flexible, cannot adapt to runtime conditions

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application consists of several operators:
1. `GenSignalOp`: Generates sequential integer values
2. `ProcessSignalOp`: Processes the input value with a simulated delay
3. `ExecutionThrottlerOp`: Controls the flow of messages between operators with different processing rates
4. `DetectEventOp`: Detects events (odd numbers) with a longer processing time
5. `ReportGenOp`: Generates reports for detected events
6. `VisualizeOp`: Visualizes the processed data

The application uses an EventBasedScheduler with multiple worker threads to enable parallel processing of the visualization and event detection paths.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/streaming_with_monitor/cpp/streaming_with_monitor
```

## Python API

The application demonstrates the same flow control patterns using the Python API. It consists of several operators:
1. `GenSignalOp`: Generates sequential integer values
2. `ProcessSignalOp`: Processes the input value with a simulated delay
3. `ExecutionThrottlerOp`: Controls the flow of messages between operators with different processing rates
4. `DetectEventOp`: Detects events (odd numbers) with a longer processing time
5. `ReportGenOp`: Generates reports for detected events
6. `VisualizeOp`: Visualizes the processed data

The application uses an EventBasedScheduler with multiple worker threads to enable parallel processing of the visualization and event detection paths.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/streaming_with_monitor/python/streaming_with_monitor.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op` decorator pattern. Both approaches (class-based and decorator-based) achieve the same functionality.