# Custom CUDA kernel Multi

Example shows ingestion of a multiple custom CUDA kernels in Holoscan SDK.

This sample has multiple custom CUDA kernels ingested into the workflow. In the application, input in the workflow comes through video_replayer operator, there are 2 instances of format_converter operator that converts the format of the input data for 2 custom CUDA kernels. Inference processor operator then ingests both the tensors from format converter operators, and applies custom CUDA kernels as per the specifications in the workflow, the result is then sent to Holoviz operator for display. 

## Data

The following dataset is used by this example:
[📦️ (NGC) Sample RacerX Video Data](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_racerx_video/files?version=20231009).

## Requirements

- CUDA 12.6+ runtime driver compatibility is required to run pre-compiled example binaries available in Holoscan SDK container or Debian
package distributions. Use the Holoscan SDK development container on a supported Holoscan platform to ensure compatibility.

## C++ Run instructions

* **from NGC container**:
  ```bash
  ./examples/custom_cuda_kernel_multi_sample/cpp/custom_cuda_kernel_multi_sample
  ```
* **source (dev container)**:
  ```bash
  ./run launch # optional: append `install` for install tree
  ./examples/custom_cuda_kernel_multi_sample/cpp/custom_cuda_kernel_multi_sample
  ```
* **source (local env)**:
  ```bash
  cd ${BUILD_OR_INSTALL_DIR}
  ./examples/custom_cuda_kernel_multi_sample/cpp/custom_cuda_kernel_multi_sample
  ```