# Bring Your Own Model

This example shows how to run inference with Holoscan and provides a mechanism, to replace the existing identity model with another model.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html) for step-by-step documentation of this example.*

## Data

The following datasets are used by this example:
- [📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).
- [(Git) Identity ONNX model](model/identity_model.onnx)

## Run instructions

The following instructions shows how to run the byom example.  If you run this example as is, it
will look similar to the `video_replayer` example.  To get the most from this example, the instructions
in this [section](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html) will walk you
through how to modify the python example code to run the application with an ultrasound segmentation model.

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download `model` and `python` folders below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/python/byom.py
  ```
* **using deb package install**:
  ```bash
  /opt/nvidia/holoscan/examples/download_example_data
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  # Need to enable write permission in the model directory to write the engine file (use with caution)
  sudo chmod a+w /opt/nvidia/holoscan/examples/bring_your_own_model/model
  python3 /opt/nvidia/holoscan/examples/bring_your_own_model/python/byom.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/bring_your_own_model/python/byom.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/bring_your_own_model/python/byom.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/bring_your_own_model/python/byom.py
  ```
