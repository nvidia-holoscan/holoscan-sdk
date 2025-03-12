# Clock Resource Example

This example demonstrates a simple ping application where the scheduler's clock attribute is used to sleep until a specified time and to measure a time interval. Timestamps are printed to the terminal to demonstrate the
behavior of these clock commands.

There are two operators involved in this example:
  1. a transmitter, set to transmit a sequence of integers from 1-3 to it's 'out' port
  2. a receiver that prints the received values to the terminal then demonstrates use of clock methods to:
    - retrieve the current time in seconds
    - retrieve the current timestamp in nanoseconds
    - sleep the compute method for a specified duration
    - sleep the compute method until a specified timestamp is reached

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/resources/clock/cpp/ping_clock
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/resources/clock/cpp/ping_clock
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/resources/clock/cpp/ping_clock
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/ping_clock.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/resources/clock/python/ping_clock.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/resources/clock/python/ping_clock.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/resources/clock/python/ping_clock.py
  ```
