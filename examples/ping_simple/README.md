# Ping Simple

This example demonstrates a simple ping application with two operators connected using add_flow().

There are two operators involved in this example:
  1. a transmitter, set to transmit a sequence of integers from 1-10 to it's 'out' port
  2. a receiver that prints the received values to the terminal

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_simple/cpp/ping_simple
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_simple/cpp/ping_simple
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_simple/cpp/ping_simple
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/ping_simple.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/ping_simple/python/ping_simple.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/ping_simple/python/ping_simple.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/ping_simple/python/ping_simple.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/ping_simple/python/ping_simple.py
  ```
