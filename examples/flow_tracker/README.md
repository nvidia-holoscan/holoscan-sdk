# Flow Tracker Example

This example demonstrates data flow tracking for an application graph with 2 root operators, 2 leaf operators, and 3 paths from the roots to the leaves. It also demonstrates the `add_probe_operator` function to track latency and message counts for intermediate operators.

![Flow Tracker Example](flow_tracker_example.png)

The example showcases two key aspects of data flow tracking:

1. **Application-level tracking**: Using the `DataFlowTracker` API to monitor end-to-end latencies and message counts across the entire application graph.
2. **Operator-level tracking**: Using the `get_data_flow_tracking_label()` API within the `PingMxOp` operator to access message label information during compute, including:
   - Number of paths the message has traversed
   - Path names (comma-separated list of operators)
   - End-to-end latency for each path in milliseconds

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/flow_tracking.html) to learn more about the Data Flow Tracking feature.*

## C++ Run instructions

- **using deb package install or NGC container**:

  ```bash
  /opt/nvidia/holoscan/examples/flow_tracker/cpp/flow_tracker
  ```

- **source (dev container)**:

  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/flow_tracker/cpp/flow_tracker
  ```

- **source (local env)**:

  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/flow_tracker/cpp/flow_tracker
  ```

## Python Run instructions

- **using python wheel**:

  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/flow_tracker.py
  ```

- **from NGC container**:

  ```bash
  python3 /opt/nvidia/holoscan/examples/flow_tracker/python/flow_tracker.py
  ```

- **source (dev container)**:

  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/flow_tracker/python/flow_tracker.py
  ```

- **source (local env)**:

  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/flow_tracker/python/flow_tracker.py
  ```
