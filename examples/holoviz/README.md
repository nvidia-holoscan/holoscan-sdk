# HolovizOp usage

Includes two examples one showing how to use the [geometry layer and the image laye](#holovizop-geometry-layer-usage) and on showing how to use the [geometry layer with 3d primitives](#holovizop-3d-geometry-layer-usage).

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/visualization.html) to learn more about Visualization in Holoscan.*

## HolovizOp geometry layer usage

As for `example/tensor_interop/python/tensor_interop.py`, this application demonstrates interoperability between a native operator (`ImageProcessingOp`) and two operators (`VideoStreamReplayerOp` and `HolovizOp`) that wrap existing C++-based operators. This application also demonstrates two additional aspects:
- capability to add multiple tensors to the message sent on the output port of a native operator (`ImageProcessingOp`)
- expected tensor shapes and arguments needed for HolovizOp in order to display overlays of various geometric primitives onto an underlying color video.

### Data

The following dataset is used by this example:
[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data/files?version=20230128).

### Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install numpy
  python3 -m pip install cupy-cuda11x # Append `-f https://pip.cupy.dev/aarch64` on aarch64
  python3 <APP_DIR>/holoviz_geometry.py
  ```
* **using deb package install**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  python3 -m pip install numpy
  python3 -m pip install cupy-cuda11x # Append `-f https://pip.cupy.dev/aarch64` on aarch64
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
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
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/holoviz/python/holoviz_geometry.py
  ```

## HolovizOp 3D geometry layer usage

As for `example/tensor_interop/python/tensor_interop.py`, this application demonstrates interoperability between a native operator (`Geometry3dOp`) and an operator (`HolovizOp`) that wraps an existing C++-based operator. This application also demonstrates two additional aspects:
- capability to add multiple tensors to the message sent on the output port of a native operator (`Geometry3dOp`)
- expected tensor shapes and arguments needed for HolovizOp in order to display overlays of various geometric primitives.

### Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install numpy
  python3 <APP_DIR>/holoviz_geometry_3d.py
  ```
* **using deb package install**:
  ```bash
  python3 -m pip install numpy
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/holoviz/python/holoviz_geometry_3d.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/holoviz/python/holoviz_geometry_3d.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/holoviz/python/holoviz_geometry_3d.py
  ```
* **source (local env)**:
  ```bash
  python3 -m pip install numpy
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/holoviz/python/holoviz_geometry_3d.py
  ```
