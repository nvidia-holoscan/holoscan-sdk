# Sequential Execution with Subgraph

This example demonstrates sequential execution control flow using a Subgraph to encapsulate
part of the execution sequence. It shows how to expose execution control interfaces from
operators within a subgraph, enabling hierarchical composition of control flow.

## Application Overview

The application creates a sequential workflow: `node1 -> node2 -> node3 -> node4`, where
`node2` and `node3` are encapsulated within a `SequentialSubgraph`. The subgraph exposes:

- An input execution interface port (`exec_in`) connected to `node2`'s input execution port
- An output execution interface port (`exec_out`) connected to `node3`'s output execution port

This allows the subgraph to be treated as a single unit in the control flow, while internally
maintaining the sequential execution of `node2` followed by `node3`.

## Key Concepts

- **Execution Control Flow**: Controls the order of operator execution without data transfer
- **Subgraph with Execution Interface Ports**: Encapsulates operators and exposes their
  execution ports as interface ports
- **Hierarchical Control Flow Composition**: Enables building complex execution patterns
  by composing subgraphs

## Expected Behavior

The operators execute in sequence:

```text
I am here - node1
I am here - node2
I am here - node3
I am here - node4
```

This demonstrates that the control flow correctly passes through the subgraph's interface
ports to the internal operators.

## Building and Running

### C++ API

#### Build instructions

Built with the SDK, see instructions from the top level README.

#### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/flow_control/sequential_with_subgraph/cpp/sequential_with_subgraph
```

### Python API

The application demonstrates the same flow control patterns using the Python API. It consists of
the same operators and subgraphs described for the C++ case above.

#### Build instructions

Built with the SDK, see instructions from the top level README.

#### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/flow_control/sequential_with_subgraph/python/sequential_with_subgraph.py
```

Note: The example also demonstrates an alternative way to define operators using the `@create_op`
decorator pattern. Both approaches (class-based and decorator-based) achieve the same
functionality.
