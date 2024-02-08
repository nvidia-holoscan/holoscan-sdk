# Tensor interoperability (GXF Tensor/DLPack/array interface)

## C++ API

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#interoperability-with-gxf-operators-cpp) to learn more about interoperability with GXF.*

This application demonstrates interoperability between a native operator (`ProcessTensorOp`) and two GXF Codelets (`SendTensor` and `ReceiveTensor`).
- The input and output ports are of type `holoscan::gxf::Entity` so that this operator can talk directly to the GXF codelets which send/receive GXF entities.
- The input/output of the entity has a tensor (`holoscan::Tensor` which is converted to `holoscan::gxf::Tensor` object inside the entity) which is used by the native operator to perform some computation and then the output tensor (in a new entity) is sent to the `ReceiveTensor` operator (codelet).
- The `ProcessTensorOp` operator uses the method in `holoscan::gxf::Tensor` to access the tensor data and perform some processing (multiplication by two) on the tensor data.
- The `ReceiveTensor` codelet gets the tensor from the entity and prints the tensor data to the terminal.

Notably, the two GXF codelets have not been wrapped as Holoscan operators, but are instead registered at runtime in the `compose` method of the application.

### Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/tensor_interop/cpp/tensor_interop
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/tensor_interop/cpp/tensor_interop
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/tensor_interop/cpp/tensor_interop
  ```

## Python API

This application demonstrates interoperability between a native operator (`ImageProcessingOp`) and two operators (`VideoStreamReplayerOp` and `HolovizOp`) that wrap existing C++-based operators using GXF Tensors, through the Holoscan Tensor object (`holoscan.core.Tensor`).
- The Holoscan Tensor object is used to get the tensor data from the GXF Entity (`holoscan::gxf::Entity`) and perform some image processing (time-varying Gaussian blur) on the tensor data.
- The output tensor (in a new entity) is sent to the `HolovizOp` operator (codelet) which gets the tensor from the entity and displays the image in the GUI. The `VideoStreamReplayerOp` operator is used to replay the video stream from the sample data.
- The Holoscan Tensor object is interoperable with DLPack or array interfaces.

*Visit the [SDK User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/holoscan_create_operator.html#interoperability-between-wrapped-and-native-python-operators) to learn more about interoperability with C++ operators.*

### Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

### Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download example .py and .yaml file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install cupy-cuda12x
  python3 <APP_DIR>/tensor_interop.py
  ```
* **using deb package install**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  python3 -m pip install cupy-cuda12x
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/tensor_interop/python/tensor_interop.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/tensor_interop/python/tensor_interop.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/tensor_interop/python/tensor_interop.py
  ```
* **source (local env)**:
  ```bash
  python3 -m pip install cupy-cuda12x
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/tensor_interop/python/tensor_interop.py
  ```
