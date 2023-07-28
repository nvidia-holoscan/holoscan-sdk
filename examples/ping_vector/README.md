# Ping Vector

This example builds on top of the ping_any example and demonstrates how to pass a vector of integers
(`std::vector<int>`) via the input and output ports. It also shows how multiple `std::vector` inputs
are received in a receiver.

There are three operators involved in this example:
  1. a transmitter, set to transmit a `vector` of 5 consecutive integers starting with `index_`.
  2. a middle operator that prints the received values, multiplies by two scaler to generate 
  two vectors. Then, it transmits the received input to transmit in its two input ports, and
  the multiplied values to the receivers.
  3. a receiver that prints all the vectors to the terminal

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_vector/cpp/ping_vector
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_vector/cpp/ping_vector
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_vector/cpp/ping_vector
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/ping_vector.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/ping_vector/python/ping_vector.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/ping_vector/python/ping_vector.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/ping_vector/python/ping_vector.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/ping_vector/python/ping_vector.py
  ```
