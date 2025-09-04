# MatX Examples

This folder contains examples demonstrating how to use the [MatX](https://github.com/NVIDIA/MatX) library within the Holoscan SDK. MatX is a high-performance C++17 library for numerical computing, especially for NVIDIA GPUs.

## Overview

The MatX examples are designed to show how to integrate MatX with Holoscan for computationally intensive tasks. MatX provides a high-level interface for creating and manipulating tensors, which can be executed on the GPU or CPU for high performance.

## Examples

### [MatX Basic](./matx_basic)
- **C++**: [`matx_basic.cu`](./matx_basic/cpp/matx_basic.cu)

This example provides a simple introduction to using MatX within a Holoscan application. It shows how to create a MatX tensor, pass it between operators using zero-copy conversion to a `holoscan::Tensor`, and perform computations on the GPU.

### [Best Practices to integrate external libraries into Holoscan pipelines --- Integrate MatX library](https://github.com/nvidia-holoscan/holohub/tree/main/tutorials/integrate_external_libs_into_pipeline#integrate-matx-library)

This tutorial in [HoloHub](https://github.com/nvidia-holoscan/holohub) shows some examples of how to use MatX in a Holoscan pipeline.

## Key Concepts Demonstrated

- **MatX Tensors**: Creating and manipulating `matx::tensor` objects.
- **Integration with Holoscan**: Using MatX within Holoscan operators for building pipelines.
- **Zero-Copy Interoperability**: Exchanging data between MatX and Holoscan tensors without copies via DLPack.

## Getting Started

Begin by exploring the [MatX Quick Start](https://nvidia.github.io/MatX/quickstart.html) guide to grasp the core concepts, then examine the [MatX Basic](./matx_basic) example to observe the API in practice. For comprehensive build and execution instructions, refer to the README file within that directory.
