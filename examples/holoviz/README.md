# HolovizOp geometry layer usage

As for `example/tensor_interop/python/tensor_interop.py`, this application demonstrates interoperability between a native operator (`ImageProcessingOp`) and two operators (`VideoStreamReplayerOp` and `HolovizOp`) that wrap existing C++-based operators. This application also demonstrates two additional aspects:
- capability to add multiple tensors to the message sent on the output port of a native operator (`ImageProcessingOp`)
- expected tensor shapes and arguments needed for HolovizOp in order to display overlays of various geometric primitives onto an underlying color video.

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data/files?version=20230128).

## Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install numpy
  python3 -m pip install cupy-cuda11x # Append `-f https://pip.cupy.dev/aarch64` on aarch64
  export HOLOSCAN_SAMPLE_DATA_PATH=<DATA_DIR>
  python3 <APP_DIR>/holoviz_geometry.py
  ```
* **using deb package install**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  python3 -m pip install numpy
  python3 -m pip install cupy-cuda11x # Append `-f https://pip.cupy.dev/aarch64` on aarch64
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  export HOLOSCAN_SAMPLE_DATA_PATH=<DATA_DIR>
  python3 /opt/nvidia/holoscan/examples/holoviz/python/holoviz_geometry.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/holoviz/python/holoviz_geometry.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/holoviz/python/holoviz_geometry.py
  ```
* **source (local env)**:
  ```bash
  python3 -m pip install numpy
  python3 -m pip install cupy-cuda11x # Append `-f https://pip.cupy.dev/aarch64` on aarch64
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_SAMPLE_DATA_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/holoviz/python/holoviz_geometry.py
  ```
