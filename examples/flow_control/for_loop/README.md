# For Loop Flow Example

This example demonstrates how to implement a for-loop pattern using dynamic flow control in Holoscan SDK. The application shows how to create a workflow where operators can be triggered multiple times in a controlled sequence. Consider the geometry of the application used in this example:

```
     <|start|>
         |
       node1
       / ^  \
   node2 |  node4
     |   |
   node3 |
     |   |(loop)
     \__/
```

The example demonstrates several key flow control patterns:
1. **Dynamic Flow Control**: Using dynamic flows to control operator execution paths
2. **Cyclic Execution**: Implementing a controlled loop pattern with fixed iterations
3. **Conditional Branching**: Switching between loop continuation and termination
4. **State Tracking**: Using metadata to track iteration count

The key aspects of this application include:
- Use of dynamic flows to implement loop behavior
- Metadata tracking for loop iteration count
- Conditional flow control based on iteration state
- Clean loop termination after desired iterations

The application demonstrates:
- How to implement for-loop patterns using dynamic flows
- How to track and manage loop iterations
- How to conditionally terminate loops
- How to use metadata for state management

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application consists of several operators:
1. `SimpleOp`: A basic operator that tracks its execution count and displays its state
2. The operators are named node1 through node4 and form a loop structure where:
   - node1 is the entry point and decision maker
   - node2 and node3 form the loop body
   - node4 is the exit path

The application uses dynamic flows to control the execution sequence and implements a loop pattern that runs for 3 iterations before exiting.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/for_loop/cpp/for_loop_execution
```

## Python API

The application demonstrates the same flow control patterns using the Python API. It consists of the same operator structure as the C++ version, implementing a controlled loop pattern.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/for_loop/python/for_loop_execution.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op` decorator pattern. Both approaches (class-based and decorator-based) achieve the same functionality.
