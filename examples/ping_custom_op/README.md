# Ping Custom Op

This example builds on top of the ping_simple example to demonstrate how to create your own custom operator.

There are three operators involved in this example (tx -> mx -> rx):
   1. a transmitter, set to transmit a sequence of integes from 1-10 to it's 'out' port
   2. a middle operator that prints the received values, multiplies by a scalar and transmits the modified values
   3. a receiver that prints the received values to the terminal

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_custom_op/cpp/ping_custom_op
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_custom_op/cpp/ping_custom_op
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_custom_op/cpp/ping_custom_op
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/ping_custom_op.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/ping_custom_op/python/ping_custom_op.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/ping_custom_op/python/ping_custom_op.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/ping_custom_op/python/ping_custom_op.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/ping_custom_op/python/ping_custom_op.py
  ```
