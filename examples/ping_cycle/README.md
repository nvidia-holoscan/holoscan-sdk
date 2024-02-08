# Ping Cycle

This example builds on top of the ping_custom_op example to demonstrate how to create a graph with a
cycle.

An operator `PingMxOp` is used three times to create a cycle in the graph. The operator receives a
value, multiplies the value by a scalar, prints the multiplied value and transmits it to the next
operator. The input port of the operator has a `holoscan::ConditionType::kNone` and does not need an
input value to trigger the operator. Therefore, the operator starts with a default value of `1` at
the beginning of the execution.

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_cycle/cpp/ping_cycle
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_cycle/cpp/ping_cycle
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_cycle/cpp/ping_cycle
  ```
