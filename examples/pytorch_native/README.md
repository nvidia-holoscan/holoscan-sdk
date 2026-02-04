# PyTorch Native

This example demonstrates using PyTorch tensors in a Holoscan Python application. It creates two tensors (on GPU if available), performs a matrix multiplication, and prints a small summary of the result.

## Installing PyTorch

PyTorch is included in the [Holoscan build container][dockerfile] and the [NGC dev container][container], providing the libtorch library needed for the inference operator.

To use PyTorch outside of these containers, install the `torch` wheel in your environment (refer to our [source Dockerfile][dockerfile] for compatible versions and indexes)

[dockerfile]: https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/Dockerfile
[container]: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan

## Run instructions

* **using python wheel**:
  ```bash
  # [Prerequisite] Download example .py file below to `APP_DIR`
  # [Prerequisite] Install PyTorch with CUDA support (see instructions above)
  # [Optional] Start the virtualenv where holoscan is installed
  python3 <APP_DIR>/matmul.py
  ```

* **from NGC container**:

  ```bash
  python3 /opt/nvidia/holoscan/examples/pytorch_native/matmul.py
  ```

- **source (dev container)**:

  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/pytorch_native/matmul.py
  ```

- **source (local env)**:

  ```bash
  # [Prerequisite] Install PyTorch with CUDA support (see instructions above)
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  python3 ${BUILD_OR_INSTALL_DIR}/examples/pytorch_native/matmul.py
  ```
