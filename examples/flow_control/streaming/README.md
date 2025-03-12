# Streaming Example with Dynamic Flow Control and Cyclic Execution

This example demonstrates how to implement dynamic flow control and cyclic execution patterns using Holoscan SDK. The application shows how to create a workflow where operators can be triggered multiple times in a controlled manner. Consider the geometry of the application used in this example:

```
                   (cycle)
                   -------
                   \     /
     <|start|>  ->  node1  ->  node2  ->  node3  ->  node4
             (triggered 5 times)
```

The example demonstrates several key flow control patterns:
1. **Dynamic Flow Control**: Using dynamic flows to control operator execution
2. **Cyclic Execution**: Implementing a cyclic flow pattern with controlled iterations
3. **Multi-stage Processing**: Processing data through multiple operators in sequence
4. **Controlled Termination**: Stopping the cycle after a specific number of iterations

The key aspects of this application include:
- Use of dynamic flows to control operator execution
- Implementation of cyclic patterns with controlled iterations
- Explicit port naming for complex flow control
- Multi-stage data processing pipeline

The application demonstrates:
- How to implement cyclic execution patterns
- How to use dynamic flows for complex routing
- How to manage operator execution sequences
- How to control iteration counts in cyclic flows

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application consists of several operators:
1. `PingTxOp`: Transmits sequential integer values
2. `PingMxOp`: Middle operator that receives and forwards values
3. `PingRxOp`: Receives and displays the final values

The application uses dynamic flows to control the execution sequence and implements a cyclic pattern that runs for 5 iterations.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/streaming/cpp/streaming_execution
```

## Python API

The application demonstrates the same flow control patterns using the Python API. It consists of several operators:
1. `PingTxOp`: Transmits sequential integer values
2. `PingMxOp`: Middle operator that receives and forwards values
3. `PingRxOp`: Receives and displays the final values

The application uses dynamic flows to control the execution sequence and implements a cyclic pattern that runs for 5 iterations.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/streaming/python/streaming_execution.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op` decorator pattern. Both approaches (class-based and decorator-based) achieve the same functionality.
