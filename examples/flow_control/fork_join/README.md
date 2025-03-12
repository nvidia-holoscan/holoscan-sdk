# Fork-Join Example

This example demonstrates how to implement a fork-join pattern using Holoscan SDK's dynamic flow control capabilities. The application shows how to create a workflow where operations can be executed in parallel and then synchronized at a join point. Consider the geometry of the application used in this example:

```
                <|start|>
                    |
                  node1
         /    /     |     \     \
      node2 node3 node4 node5 node6
        \     \     |     /     /
         \     \    |    /     /
          \     \   |   /     /
           \     \  |  /     /
                  node7
                    |
                  node8
```

The example demonstrates several key flow control patterns:
1. **Fork Pattern**: Splitting workflow into multiple parallel paths
2. **Join Pattern**: Synchronizing multiple paths back into a single flow
3. **Parallel Processing**: Running multiple operators concurrently
4. **Flow Synchronization**: Managing concurrent execution paths

The key aspects of this application include:
- Multi-threaded execution of parallel operators
- Synchronization of parallel paths at join points
- Dynamic flow control using add_flow
- Configurable worker thread pool

The application demonstrates:
- How to implement fork-join patterns
- How to manage parallel execution paths
- How to synchronize multiple operators
- How to configure multi-threaded execution

*Visit the [Dynamic Flow Control section of the SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_dynamic_flow_control.html) to learn more about flow control patterns.*

## C++ API

The application consists of a single operator type that is instantiated multiple times:
1. `SimpleOp`: A basic operator that logs its execution and simulates processing with a delay

The application uses an EventBasedScheduler with multiple worker threads to enable parallel processing of the forked paths.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/fork_join/cpp/fork_join_execution
```

## Python API

The application demonstrates the same fork-join patterns using the Python API. It uses the same operator structure:
1. `SimpleOp`: A basic operator that logs its execution and simulates processing with a delay

The application uses an EventBasedScheduler with multiple worker threads to enable parallel processing of the forked paths.

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/fork_join/python/fork_join_execution.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op` decorator pattern. Both approaches (class-based and decorator-based) achieve the same functionality.
