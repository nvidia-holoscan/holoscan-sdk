# Ping Conditional

This example builds on top of the ping_any example and demonstrates how to pass a raw pointer
pointer to an integer (`int*`) between the operators.

There are three operators involved in this example:
  1. a transmitter, set to transmit a raw pointer to integer when odd values are sent and `nullptr`
  when even values are sent. The transmitter also has a `CountCondition` to trigger for 10 times.
  2. a middle operator that prints the received values, when a valid pointer is received and transmits the integer value at the integer pointer to the receiver. When the middle operator's input is a `nullptr`, it just prints a counter value.
  3. a receiver that prints the received integer value to the terminal. The receiver's input has a `Condition` to trigger even when there is no input. The receiver also has a `CountCondition` to trigger for 20 times whereas the transmitter triggers for 10 times. The receiver checks whether its input has any data with `received_value.has_value()`.

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_conditional/cpp/ping_conditional
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_conditional/cpp/ping_conditional
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_conditional/cpp/ping_conditional
  ```