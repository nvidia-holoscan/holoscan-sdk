# Holoscan SDK Core modules

Core modules under this directory extend Holoscan SDK core APIs with
domain-specific functionality. See the [module development guide](DEVELOPER.md)
for the common contributor layout, package-boundary, documentation, and test
requirements.

| Module | Overview | Primary Backend |
| --- | --- | --- |
| [Inference](inference/) | Simplified inference interface for Holoscan pipelines | Holoinfer library based on NVIDIA TensorRT |
| [Vision](vision/) | Off-the-shelf computer vision components for Holoscan pipelines | NVIDIA CV-CUDA library |
| [Visualization](visualization/) | GPU-ready, high-performance rendering for Holoscan pipelines | Holoviz custom visualization library |

User-facing documentation for these modules lives in the [Holoscan Modules](https://docs.nvidia.com/holoscan/sdk-user-guide/modules) section of the user guide.

Refer to the [Holoscan Ecosystem discovery page](https://nvidia-holoscan.github.io/holohub/modules/) to learn about additional Holoscan Modules.
