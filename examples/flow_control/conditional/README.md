# Conditional Flow Example

This example demonstrates how to implement conditional flow control using Holoscan SDK's dynamic flow capabilities. The application shows how to create a workflow where data is routed differently based on runtime conditions. Consider the geometry of the application used in this example:

```
                   node1 (launch twice)
                   /   \
               node2   node4
                 |       |
               node3   node5
```

The example demonstrates several key flow control patterns:
1. **Conditional Flow**: Dynamic routing of data based on runtime conditions
2. **Multiple Paths**: Handling different execution paths based on conditions
3. **Controlled Iterations**: Using CountCondition to limit operator execution
4. **Dynamic Flow Control**: Runtime decision-making for data routing

The key aspects of this application include:
- Use of dynamic flows for conditional routing
- Implementation of controlled iterations using CountCondition
- Multiple execution paths based on runtime values
- Simple operator state management

The application demonstrates:
- How to implement conditional routing logic
- How to use CountCondition for iteration control
- How to manage multiple execution paths
- How to use operator state to influence routing decisions

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application uses multiple instances of the following operator:
1. `SimpleOp`: A basic operator that maintains a counter and prints its execution
- Launched with CountCondition(2) for node1
- Routes data to either node2 or node4 based on counter value
- Subsequent nodes process and display execution status

The application uses dynamic flows to control the execution sequence and implements conditional routing based on the operator's internal state.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/conditional/cpp/conditional_execution
```

## Python API

The application demonstrates the same flow control patterns using the Python API. It consists of the same operators as the C++ version:
1. `SimpleOp`: A basic operator that maintains a counter and prints its execution
- Launched with CountCondition(2) for node1
- Routes data to either node2 or node4 based on counter value
- Subsequent nodes process and display execution status

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/conditional/python/conditional_execution.py
```

Note: The example demonstrates an alternative way to define operators using the `@create_op` decorator pattern. Both approaches (class-based and decorator-based) achieve the same functionality.
