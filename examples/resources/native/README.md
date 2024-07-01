# Native Resource Example
``
This example demonstrates how a C++ `holoscan::Resource` (or Python `holoscan.core.Resource`) can be created without wrapping an underlying GXF Component.

## C++ Run instructions

* **using deb package install or NGC container**:
  ```bash
  /opt/nvidia/holoscan/examples/resources/native/cpp/native_resource
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/resources/native/cpp/native_resource
  ```
* **source (local env)**:
  ```bash
  ${BUILD_OR_INSTALL_DIR}/examples/resources/native/cpp/native_resource
  ```

## Python Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/native_resource.py
  ```
* **using deb package install**:
  ```bash
  export PYTHONPATH=/opt/nvidia/holoscan/python/lib
  python3 /opt/nvidia/holoscan/examples/resources/native/python/native_resource.py
  ```
* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/resources/native/python/native_resource.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/resources/native/python/native_resource.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/resources/native/python/native_resource.py
  ```
