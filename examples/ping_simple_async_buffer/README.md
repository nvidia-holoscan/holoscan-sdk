# Ping Simple Async Buffer Example

This example demonstrates the use of `IOSpec::ConnectorType::kAsyncBuffer` connectors in a simple Holoscan ping application.

There are two operators involved in this example:
  1. a transmitter, set to transmit a sequence of integers from 1-20 to its
     'out' port. The transmitter will sleep for 10ms before emitting an integer.
  2. a receiver that prints the received values to the terminal. The receiver
     will sleep for 5ms before receiving an integer.

The application is configured to use the Event-based scheduler with 2 worker
threads. As the transmitter and receiver are connected with the
`kAsyncBuffer` connector, they will run independently in parallel in separate
threads. Roughly, the receiver is set to receive 2 messages for every message
being emitted by the transmitter because of their respective sleep durations.

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_simple_async_buffer/cpp/ping_simple_async_buffer
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_simple_async_buffer/cpp/ping_simple_async_buffer
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_simple_async_buffer/cpp/ping_simple_async_buffer
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/ping_simple_async_buffer.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/ping_simple_async_buffer/python/ping_simple_async_buffer.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/ping_simple_async_buffer/python/ping_simple_async_buffer.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/ping_simple_async_buffer/python/ping_simple_async_buffer.py
  ```
