# Conditional Routing Flow Example

This example demonstrates how to implement conditional routing using Holoscan SDK's dynamic flow control capabilities. The application shows how to dynamically route data to different operators based on runtime conditions. Consider the geometry of the application used in this example:

```
                node1 (launch twice, emitting data to 'output' port)
               /     \
             node2 node3
```

The example demonstrates several key flow control patterns:
1. **Dynamic Flow Control**: Using dynamic flows to route data based on conditions
2. **Conditional Routing**: Directing data to different operators based on values
3. **Multiple Receivers**: Handling multiple downstream operators
4. **Controlled Execution**: Using CountCondition to limit iterations

The key aspects of this application include:
- Use of dynamic flows for conditional routing
- Value-based decision making for data flow
- Multiple receiver operator handling
- Controlled number of iterations

The application demonstrates:
- How to implement conditional routing patterns
- How to use dynamic flows for value-based routing
- How to manage multiple downstream operators
- How to control execution iterations

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application consists of several operators:
1. `PingTx`: Transmits sequential integer values
2. `PingRx`: Receives and displays the values (used for both node2 and node3)

The application uses dynamic flows to route odd numbers to node2 and even numbers to node3.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/conditional_routing/cpp/conditional_routing_execution
```

## Python API

The application demonstrates the same flow control patterns using the Python API. It consists of several operators:
1. `PingTx`: Transmits sequential integer values
2. `PingRx`: Receives and displays the values (used for both node2 and node3)

The application uses dynamic flows to route odd numbers to node2 and even numbers to node3.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/conditional_routing/python/conditional_routing_execution.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op` decorator pattern. Both approaches (class-based and decorator-based) achieve the same functionality.
