# Using a Function Decorator to Build Python Operators 

This is an example of mixed use of native Python operators and wrapped C++ operators. In this example, instead of explicitly creating a Python operator from inheriting from `holoscan.core.Operator`, we instead demonstrate how `holoscan.core.decorator` can be used to decorator an existing function, turning it into an Operator.

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/video_replayer.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/python_decorator/video_replayer.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/python_decorator/video_replayer.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/python_decorator/video_replayer.py
  ```
