# Periodic Condition with Async Buffer

This example demonstrates how to use `PeriodicCondition` together with an asynchronous buffer connector to decouple two operators that run at different rates.

There are two operators involved in this example:
  1. **Transmitter** (`PingTxOp` / `PingTxCustom`) – publishes an incrementing integer on port `out`. It is configured with a `CountCondition` of **10** and a `PeriodicCondition` whose period is `tx_period_ms` (default **100 ms**).
  2. **Receiver** (`PingRxOp` / `PingRxCustom`) – prints the received values to the terminal. It is also limited by a `CountCondition` of **10** but uses its own `PeriodicCondition` with period `rx_period_ms` (default **200 ms**).

Because the two operators tick at different frequencies, an *async buffer* (`IOSpec::ConnectorType::kAsyncBuffer` / `IOSpec.ConnectorType.ASYNC_BUFFER`) is used for the edge so that the faster producer does not block on the slower consumer.

Optionally, the application can be executed with an **Event-Based Scheduler** (pass `1` as the third command-line argument) to illustrate how the same graph behaves with a cooperative scheduler that wakes operators only when their conditions evaluate to `true`. Omitting the third argument or any other value in the third argument will result in using the default greedy scheduler.

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/ping_periodic_async_buffer/cpp/ping_periodic_async_buffer [tx_period_ms] [rx_period_ms] [event_scheduler_flag]
  # Example using the defaults with an event-based scheduler
  /opt/nvidia/holoscan/examples/ping_periodic_async_buffer/cpp/ping_periodic_async_buffer
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/ping_periodic_async_buffer/cpp/ping_periodic_async_buffer
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/ping_periodic_async_buffer/cpp/ping_periodic_async_buffer
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Activate the virtual environment where Holoscan is installed
  python3 <APP_DIR>/ping_periodic_async_buffer.py [tx_period_ms] [rx_period_ms] [event_scheduler_flag]
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/ping_periodic_async_buffer/python/ping_periodic_async_buffer.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/ping_periodic_async_buffer/python/ping_periodic_async_buffer.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/ping_periodic_async_buffer/python/ping_periodic_async_buffer.py
  ```
