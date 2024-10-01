# Video Replayer

Minimal example to demonstrate the use of the video stream replayer operator to load video from disk.

The video frames need to have been converted to a gxf entity format to use as input. You can use the `convert_video_to_gxf_entities.py` script installed in `/opt/nvidia/holoscan/bin` or available [on GitHub](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/scripts#convert_video_to_gxf_entitiespy) (tensors will be loaded on the GPU).

> Note: Support for H264 stream support is in progress and can be found on [HoloHub](https://nvidia-holoscan.github.io/holohub)

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/video_replayer.html) for step-by-step documentation of this example.*

#### Note on error logged by the application
Note that it is currently expected that this application logs the following error during shutdown

```text
[error] [ucx_context.cpp:466] Connection dropped with status -25 (Connection reset by remote peer)
```

This will be logged by the worker that is running "fragment2" after "fragment1" has sent all messages. It is caused by fragment 1 starting to shutdown after its last message has been sent, resulting in severing of connections from fragment 2 receivers to fragment 1 transmitters.

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

## C++ Run instructions

* **using deb package install**:
  ```bash
  sudo /opt/nvidia/holoscan/examples/download_example_data
  export HOLOSCAN_INPUT_PATH=/opt/nvidia/holoscan/data
  ./examples/video_replayer/cpp/video_replayer
  ```
* **from NGC container**:
  ```bash
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

This example can also be run with a pair of visualization windows. In order to do that first modify
the configuration file to set `dual_window` to `false`. This can be done via:

```bash
sed -i -e 's#^dual_window:.*#dual_window: true#' ./examples/video_replayer/cpp/video_replayer.yaml
```
and then follow the same instructions as above to run the application.


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
  sudo /opt/nvidia/holoscan/examples/download_example_data
  export HOLOSCAN_INPUT_PATH=/opt/nvidia/holoscan/data
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

This example can also be run with a pair of visualization windows. In order to do that first modify
the configuration file to set `dual_window` to `true`. This can be done via:

```bash
sed -i -e 's#^dual_window:.*#dual_window: true#' ./examples/video_replayer/python/video_replayer.yaml
```
and then follow the same instructions as above to run the application.
