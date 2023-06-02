# Bring Your Own Model - Ultrasound

This example shows how to use the [Bring Your Own Model](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/clara_holoscan_applications.html#bring-your-own-model-byom-customizing-the-ultrasound-segmentation-application-for-your-model) (BYOM) concept for Holoscan by replacing the existing identity model.

## Data

The following datasets are used by this example:
- [📦️ (NGC) Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data/files?version=20230128)
- [(Git) Identity ONNX model](model/identity_model.onnx)

## Run instructions

The following instructions shows how to run the byom example.  If you run this example as is, it
will look similar to the `video_replayer` example.  To get the most from this example, the instructions
in this [section](https://docs.nvidia.com/holoscan/sdk-user-guide/examples/byom.html) will walk you
through how to modify the python example code to run the application with an ultrasound segmentation model.

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  # [Prerequisite] Download `model` and `python` folders below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  export HOLOSCAN_SAMPLE_DATA_PATH=<DATA_DIR>
  python3 <APP_DIR>/python/byom.py
  ```
* **using deb package install**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  export HOLOSCAN_SAMPLE_DATA_PATH=<DATA_DIR>
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
  export HOLOSCAN_SAMPLE_DATA_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/bring_your_own_model/python/byom.py
  ```
