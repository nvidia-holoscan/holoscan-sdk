# Mixed Flows Example

This example demonstrates how to implement mixed flow patterns using Holoscan SDK's dynamic flow control capabilities. The application shows how to create workflows where data can be conditionally routed to different operators based on runtime conditions. Consider the geometry of the application used in this example:

```
                node1 (from 'output' port)
            /     |     \
          node2 alarm  node3
```

The example demonstrates several key flow control patterns:
1. **Dynamic Flow Control**: Using dynamic flows to conditionally route data
2. **Conditional Routing**: Routing data based on value conditions
3. **Mixed Flow Patterns**: Combining different flow control patterns in one application

The key aspects of this application include:
- Dynamic routing based on message values
- Explicit port naming for complex routing
- Combination of static and dynamic flows

The application demonstrates:
- How to implement conditional routing based on data values
- How to combine multiple flow patterns in one application
- How to use explicit port naming for complex flows

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application consists of several operators:
1. `PingTx`: Transmits sequential integer values
2. `PingRx`: Receives and displays values
3. `SimpleOp`: Demonstrates basic operator functionality with name printing

The application uses dynamic flows to route even-numbered values to node3 and odd-numbered values to node2, while the alarm operator is always executed.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/mixed_flows/cpp/mixed_flows_execution
```

## Python API

The application demonstrates the same flow control patterns using the Python API. It consists of several operators:
1. `PingTx`: Transmits sequential integer values
2. `PingRx`: Receives and displays values
3. `SimpleOp`: Demonstrates basic operator functionality with name printing

The application uses dynamic flows to route even-numbered values to node3 and odd-numbered values to node2, while the alarm operator is always executed.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/mixed_flows/python/mixed_flows_execution.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op` decorator pattern. The Python implementation includes commented code that shows how to implement the same functionality using decorators instead of classes.
