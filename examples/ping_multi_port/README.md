# Ping Multi Port

This example builds on top of the ping_custom_op example and demonstrates how to send and receive data using
a custom data class.  Additionally, this example shows how create and use operators that have multiple input
and/or output ports.

There are three operators involved in this example:
  1. a transmitter, set to transmit a sequence of even integers on port `out1` and odd integers on port `out2`.
  2. a middle operator that prints the received values, multiplies by a scalar and transmits the modified values
  3. a receiver that prints the received values to the terminal

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_multi_port/cpp/ping_multi_port
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_multi_port/cpp/ping_multi_port
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_multi_port/cpp/ping_multi_port
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/ping_multi_port.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/ping_multi_port/python/ping_multi_port.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/ping_multi_port/python/ping_multi_port.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/ping_multi_port/python/ping_multi_port.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/ping_multi_port/python/ping_multi_port.py
  ```
