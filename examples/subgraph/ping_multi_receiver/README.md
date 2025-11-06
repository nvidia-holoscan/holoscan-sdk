# Subgraph Ping Multi-Receiver Example

This folder contains examples demonstrating Holoscan's subgraph functionality for creating reusable, composable operator patterns. The examples showcase how subgraphs enable modular application design by encapsulating related operators and exposing clean interfaces for external connections.

The subgraphs used in this example application are very small, but the true utility would be realized in real-world applications involving larger graphs. An example might be a multi-stage camera pre-processing pipeline subgraph. This same subgraph could be reused across multiple cameras in the same application or shared across other applications also using the same camera pre-processing logic.

## C++ Application

The C++ application demonstrates several subgraph patterns:

1. **Reusable Subgraphs**: `PingTxSubgraph` and `PingRxSubgraph` can be instantiated multiple times with different instance names
2. **Interface Port Mapping**: Subgraphs expose external ports that map to internal operator ports
3. **Multi-receiver Pattern**: `MultiPingRxSubgraph` shows how to create subgraphs that can receive from multiple sources (this just involves exposing the multi-receiver Operator port as one of the subgraphs interface ports).
4. **Qualified Naming**: Operators within subgraphs get qualified names (e.g., "tx1_transmitter", "tx2_transmitter")

### Key Subgraphs and Components

**PingTxSubgraph**: Contains a `PingTxOp` transmitter that sends 8 messages, connected with a `ForwardingOp` to demonstrate internal operator chaining.

**PingRxSubgraph**: Contains a single `PingRxOp` receiver that accepts messages from one source.

**MultiPingRxSubgraph**: Contains a `MultiPingRxOp` that can receive messages from multiple transmitters simultaneously using the multi-receiver pattern.

**ForwardingOp**: A simple pass-through operator that forwards any received message to its output port, demonstrating how subgraphs can contain multiple connected operators.

### Application Architectures

The example application illustrates the use of make_subgraph for creating subgraph instances. The usual way of using add_flow to connect operators also works for connections to subgraph interface ports. The application demonstrates flows being added from subgraph->subgraph as well as from operator->subgraph. The case of subgraph->operator works in the same way, but is not used in this application.

### C++ Run Instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/subgraph/ping_multi_receiver/cpp/subgraph_ping
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/subgraph/ping_multi_receiver/cpp/subgraph_ping
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/subgraph/ping_multi_receiver/cpp/subgraph_ping
  ```

## Python Application

The Python application provides the same functionality as the C++ version, demonstrating how subgraph concepts translate to Python using the Holoscan Python API. The Python implementation includes:

1. **Custom Operators**: `ForwardingOp` and `MultiPingRxOp` implemented as Python classes
2. **Subgraph Classes**: `PingTxSubgraph`, `PingRxSubgraph`, and `MultiPingRxSubgraph`
3. **Multi-receiver Pattern**: Demonstrates Python-specific usage of `IOSpec.ANY_SIZE` for multi-receiver ports
4. **Application Architecture**: Same connection patterns as C++ version using Python-appropriate syntax

### Python Run Instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/subgraph_ping.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/subgraph/ping_multi_receiver/python/subgraph_ping.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/subgraph/ping_multi_receiver/python/subgraph_ping.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/subgraph/ping_multi_receiver/python/subgraph_ping.py
  ```

## Key Features Demonstrated

**Modular Design**: Subgraphs allow you to create reusable components that can be instantiated multiple times within an application.

**Clean Interfaces**: Interface ports provide clean external APIs for Subgraphs, hiding internal complexity while exposing only the necessary connection points.

**Automatic Name Qualification**: The framework automatically qualifies operator names within Subgraphs to prevent naming conflicts (e.g., "tx1_transmitter", "tx2_transmitter" from instances "tx1" and "tx2").

**Flexible Connection Patterns**: Subgraphs can be connected to other Subgraphs or raw operators using the same `add_flow` API, enabling flexible application architectures.

**Multi-receiver Support**: Demonstrates how Subgraphs can encapsulate complex patterns like multi-receiver operators that accept input from multiple sources.

**Nested Subgraph Support**: Although not demonstrated in this example, it is possible for subgraphs to contain other subgraphs for even further decomposition of large applications.

## Expected Output

The application will show:
- Printing of port mapping information showing the final graph structure once all subgraphs have been composed.
- Successful message transmission and reception via logging of received message information

This example serves as a foundation for understanding how to build complex, modular Holoscan applications using reusable subgraph components.

