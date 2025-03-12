# Sequential Execution Example

This example demonstrates how to implement a simple sequential execution pattern using Holoscan SDK's implicit execution ports. The application shows how to create a basic workflow where operators execute in a defined sequence. Consider the geometry of the application used in this example:

```
<|start|> -> node1 -> node2 -> node3
```

The example demonstrates several key flow control patterns:
1. **Sequential Execution**: Processing data through a linear chain of operators
2. **Basic Flow Control**: Using start operator (`start_op()`) and connect operators (`add_flow()`) to define operator execution order
3. **Simple Operator Chain**: Implementing a straightforward operator pipeline

The key aspects of this application include:
- Basic operator chaining using implicit execution ports
- Simple linear execution pattern
- Demonstration of operator sequencing
- Basic workflow definition

The application demonstrates:
- How to implement sequential operator execution
- How to chain multiple operators together
- How to create a simple linear workflow

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application consists of multiple instances of a single operator type:
1. `SimpleOp`: A basic operator that prints its name during execution
   - Used to create three instances: node1, node2, and node3
   - Demonstrates sequential processing through the operator chain

The application uses implicit execution ports to control the execution sequence in a linear pattern.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/sequential/cpp/sequential_execution
```

## Python API

The application demonstrates the same flow control patterns using the Python API. It consists of the same operator:
1. `SimpleOp`: A basic operator that prints its name during execution
   - Used to create three instances: node1, node2, and node3
   - Demonstrates sequential processing through the operator chain

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/sequential/python/sequential_execution.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op` decorator pattern. Both approaches (class-based and decorator-based) achieve the same functionality.
