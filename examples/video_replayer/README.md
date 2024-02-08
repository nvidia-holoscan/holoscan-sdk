# Video Replayer

Minimal example to demonstrate the use of the video stream replayer operator to load video from disk. The video frames need to have been converted to a gxf entity format, as shown [here](../../scripts/README.md#convert_video_to_gxf_entitiespy).

> Note: Support for H264 stream support is in progress and can be found on [HoloHub](https://nvidia-holoscan.github.io/holohub)

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/video_replayer.html) for step-by-step documentation of this example.*

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

## C++ Run instructions

* **using deb package install**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `/opt/nvidia/data`
  cd /opt/nvidia/holoscan # to find dataset
  ./examples/video_replayer/cpp/video_replayer
  ```
* **from NGC container**:
  ```bash
  cd /opt/nvidia/holoscan # to find dataset
  ./examples/video_replayer/cpp/video_replayer
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/video_replayer/cpp/video_replayer
  ```
* **source (local env)**:
  ```bash
  cd ${BUILD_OR_INSTALL_DIR}
  ./examples/video_replayer/cpp/video_replayer
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/video_replayer.py
  ```
* **using deb package install**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/video_replayer/python/video_replayer.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/video_replayer/python/video_replayer.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/video_replayer/python/video_replayer.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/video_replayer/python/video_replayer.py
  ```
