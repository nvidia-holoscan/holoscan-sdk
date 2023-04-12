# Video Replayer

Minimal example to demonstrate the use of the video stream replayer operator to load video from disk. The video frames need to have been converted to a gxf entity format, as shown [here](../../scripts/README.md#convert_video_to_gxf_entitiespy).

> Note: Support for H264 stream support is in progress

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data).

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
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  export HOLOSCAN_SAMPLE_DATA_PATH=<DATA_DIR>
  python3 <APP_DIR>/video_replayer.py
  ```
* **using deb package install**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  export HOLOSCAN_SAMPLE_DATA_PATH=<DATA_DIR>
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
  export HOLOSCAN_SAMPLE_DATA_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/video_replayer/python/video_replayer.py
  ```
