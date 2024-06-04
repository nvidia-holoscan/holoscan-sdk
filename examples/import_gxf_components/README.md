# Importing existing GXF Codelets/Components as Holoscan Operators/Resources

## Overview

This application demonstrates how to import existing GXF Codelets and Components as Holoscan Operators/Resources. The example code is adapted from the [Tensor interoperability example](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples/tensor_interop) (`examples/tensor_interop`) and modified to illustrate the use of GXFCodeletOp and GXFComponentResource classes to import GXF Codelet/Component and customize the `setup()` and `initialize()` methods.

## C++ API

The main components include a set of custom operators that encapsulate GXF Codelets for sending and receiving tensors, and a resource class for managing block memory pools.

- **GXFSendTensorOp**: An operator that wraps a GXF Codelet responsible for sending tensors.
- **GXFReceiveTensorOp**: Extends `GXFCodeletOp` to wrap a GXF Codelet that receives tensors, with customizable setup and initialization.
- **MyBlockMemoryPool**: Represents a resource wrapping the `nvidia::gxf::BlockMemoryPool` GXF component, used to manage memory allocation.

### Run Instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/import_gxf_components/cpp/import_gxf_components
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/import_gxf_components/cpp/import_gxf_components
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/import_gxf_components/cpp/import_gxf_components
  ```

## Python API

Python example shows how to utilize the `GXFCodeletOp` and `GXFComponentResource` in a Holoscan application, focusing on video processing using custom and built-in operators.

- **ToDeviceMemoryOp**: A derived class from `GXFCodeletOp` for copying tensors to device memory.
- **ImageProcessingOp**: Processes video frames using Gaussian filtering with CuPy.
- **DeviceMemoryPool**: Manages device memory allocations.

### Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download NGC dataset above to `DATA_DIR`
  export HOLOSCAN_INPUT_PATH=<DATA_DIR>
  # [Prerequisite] Download example .py and .yaml file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 -m pip install cupy-cuda12x
  python3 <APP_DIR>/import_gxf_components.py
  ```
* **using deb package install**:
  ```bash
  /opt/nvidia/holoscan/examples/download_example_data
  export HOLOSCAN_INPUT_PATH=/opt/nvidia/holoscan/data
  python3 -m pip install cupy-cuda12x
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/import_gxf_components/python/import_gxf_components.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/import_gxf_components/python/import_gxf_components.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/import_gxf_components/python/import_gxf_components.py
  ```
* **source (local env)**:
  ```bash
  python3 -m pip install cupy-cuda12x
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/import_gxf_components/python/import_gxf_components.py
  ```
