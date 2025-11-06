# HolovizOp usage

Includes multiple examples showing how to use various features of the Holoviz operator
- use the [camera with 3d primitives](#holovizop-camera-usage)
- use the [geometry layer and the image layer](#holovizop-geometry-layer-usage)
- use the [geometry layer with 3d primitives](#holovizop-3d-geometry-layer-usage)
- use the [first pixel out and presentation done conditions](#holovizop-conditions-usage)
- use [layer views](#holovizop-views-usage)

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/visualization.html) to learn more about Visualization in Holoscan.*

## HolovizOp camera usage

This application demonstrates how to render 3d primitives and view the geometry from different camera positions and also how to retrieve the current camera position from the Holoviz operator.
The `GeometrySourceOp` generates a 3D cube, each side is output as a separate tensor. It also randomly switches between camera positions each second.
The `HolovizOp` renders the 3d primitives, smoothly interpolates between camera positions, allows to use the mouse to change the camera and outputs the current camera position for `CameraPoseRxOp`.
The `CameraPoseRxOp` receives camera pose information and prints to the console (but only once every second).

### C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/holoviz/cpp/holoviz_camera
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/holoviz/cpp/holoviz_camera
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/holoviz/cpp/holoviz_camera
  ```

## HolovizOp geometry layer usage

As for `example/tensor_interop/python/tensor_interop.py`, this application demonstrates interoperability between a native operator (`ImageProcessingOp`) and two operators (`VideoStreamReplayerOp` and `HolovizOp`) that wrap existing C++-based operators. This application also demonstrates two additional aspects:
- capability to add multiple tensors to the message sent on the output port of a native operator (`ImageProcessingOp`)
- expected tensor shapes and arguments needed for HolovizOp in order to display overlays of various geometric primitives onto an underlying color video.

### Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

### C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/holoviz/cpp/holoviz_geometry
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/holoviz/cpp/holoviz_geometry
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/holoviz/cpp/holoviz_geometry
  ```

### Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install "numpy<2.0"
  python3 <APP_DIR>/holoviz_geometry.py
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
  python3 -m pip install "numpy<2.0"
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
  python3 -m pip install "numpy<2.0"
  python3 <APP_DIR>/holoviz_geometry_3d.py
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
  python3 -m pip install "numpy<2.0"
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/holoviz/python/holoviz_geometry_3d.py
  ```

## HolovizOp views usage

This example demonstrates how to use views on layers. A layer can be an image or geometry. A view defines how a layer is placed in the output window. A view
defines the 2D offset and size of a layer and also can be placed in 3d space using a 3d transformation matrix. More information can be found [here](https://docs.nvidia.com/holoscan/sdk-user-guide/visualization.html#views).
The `VideoStreamReplayerOp` reads video frames from a file and passes them to the `ImageViewsOp`.
The `ImageViewsOp` takes the frames, defines multiple dynamic and static views and passes the video frames to the `HolovizOp`. The `ImageViewsOp` also generates
the view and data for a rotating frame counter.
The `HolovizOp` renders the video frame views and the frame counter.

### Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

### Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install "numpy<2.0"
  python3 <APP_DIR>/holoviz_views.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/holoviz/python/holoviz_views.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/holoviz/python/holoviz_views.py
  ```
* **source (local env)**:
  ```bash
  python3 -m pip install "numpy<2.0"
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/holoviz/python/holoviz_views.py
  ```

## HolovizOp conditions usage

This example demonstrates the use of Holoviz's synchronization conditions, including the FirstPixelOut and PresentDone conditions. These conditions allow you to control the execution flow of the operator, synchronizing data presentation and frame rendering with the graphics subsystem (first pixel out and present events). It shows how to wire up HolovizOp with these custom conditions and track readiness for presenting new frames.

### Command Line Options

The example supports the following command line options:

* `-h, --help` - Display help information
* `-c, --count <N>` - Limit the number of frames to display before the application exits. Default is -1 (unlimited). Any positive integer will limit the number of frames displayed.
* `-t, --type <TYPE>` - Select the condition type to use for synchronization. Default is `present_done`.
  * `first_pixel_out` - Use FirstPixelOutCondition to wait for the first pixel out signal from the display
  * `present_done` - Use PresentDoneCondition to wait for presentation completion
* `-v, --vsync` - Enable vsync for the visualizer. Default is disabled.

### C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/holoviz/cpp/holoviz_conditions
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/holoviz/cpp/holoviz_conditions
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/holoviz/cpp/holoviz_conditions
  ```
