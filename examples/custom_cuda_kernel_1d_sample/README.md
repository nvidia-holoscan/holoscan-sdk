# Custom CUDA kernel 1D

Example shows ingestion of a single 1D custom CUDA kernel.

In the example, input in the workflow comes through video_replayer operator, format_converter operator then converts the format of the input data. Inference processor operator then ingests the formatted input and applies custom CUDA kernels and the result is then sent to Holoviz operator for display. 

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

## Requirements

- CUDA 12.6+ runtime driver compatibility is required to run pre-compiled example binaries available in Holoscan SDK container or Debian
package distributions. Use the Holoscan SDK development container on a supported Holoscan platform to ensure compatibility.

## C++ Run instructions

* **from NGC container**:
  ```bash
  ./examples/custom_cuda_kernel_1d_sample/cpp/custom_cuda_kernel_1d_sample
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/custom_cuda_kernel_1d_sample/cpp/custom_cuda_kernel_1d_sample
  ```
* **source (local env)**:
  ```bash
  cd ${BUILD_OR_INSTALL_DIR}
  ./examples/custom_cuda_kernel_1d_sample/cpp/custom_cuda_kernel_1d_sample
  ```

## Python Run instructions

* **from NGC container**:
  ```bash
  python3 /opt/nvidia/holoscan/examples/custom_cuda_kernel_1d_sample/python/custom_cuda_kernel_1d_sample.py
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  python3 ./examples/custom_cuda_kernel_1d_sample/python/custom_cuda_kernel_1d_sample.py
  ```
* **source (local env)**:
  ```bash
  export PYTHONPATH=${BUILD_OR_INSTALL_DIR}/python/lib
  export HOLOSCAN_INPUT_PATH=${SRC_DIR}/data
  python3 ${BUILD_OR_INSTALL_DIR}/examples/custom_cuda_kernel_1d_sample/python/custom_cuda_kernel_1d_sample.py
  ```