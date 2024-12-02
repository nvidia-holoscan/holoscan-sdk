# Ping Simple

This example demonstrates a simple ping application with two operators connected using add_flow().

There are two operators involved in this example:
  1. a transmitter, set to transmit a sequence of integers from 1-10 to it's 'out' port
  2. a receiver that prints the received values to the terminal

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/resources/thread_pool/cpp/ping_simple_thread_pool
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/resources/thread_pool/cpp/ping_simple_thread_pool
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/resources/thread_pool/cpp/ping_simple_thread_pool
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/ping_simple_thread_pool.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/resources/thread_pool/python/ping_simple_thread_pool.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/resources/thread_pool/python/ping_simple_thread_pool.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/resources/thread_pool/python/ping_simple_thread_pool.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/resources/thread_pool/python/ping_simple_thread_pool.py
  ```
