(relevant-technologies)=

# Relevant Technologies

Holoscan accelerates streaming AI applications by leveraging both hardware and software.
The Holoscan SDK relies on multiple core technologies to achieve low latency and high throughput:

- {ref}`gpudirect_rdma`
- {ref}`gxf-tech`
- {ref}`tensorrt`
- {ref}`cuda_rendering_interop`
- {ref}`npp`
- {ref}`ucx`
- {ref}`matx`

(gpudirect_rdma)=

## Rivermax and GPUDirect RDMA

The NVIDIA Developer Kits equipped with a [ConnectX network adapter](https://www.nvidia.com/en-us/networking/ethernet-adapters/) can be used along with the [NVIDIA Rivermax SDK](https://developer.nvidia.com/networking/rivermax) to provide an extremely efficient network connection that is further optimized for GPU workloads by using [GPUDirect](https://developer.nvidia.com/gpudirect) for RDMA. This technology avoids unnecessary memory copies and CPU overhead by copying data directly to or from pinned GPU memory, and supports both the integrated GPU or the discrete GPU.

:::{note}
NVIDIA is committed to supporting hardware vendors enabling RDMA within their own drivers, an example of which is provided by the {ref}`aja_video_systems`, as part of a partnership with
NVIDIA for the Holoscan SDK. The [AJASource operator](https://github.com/nvidia-holoscan/holohub/tree/holoscan-sdk-3.3.0/operators/aja_source) is an example of how the SDK can leverage RDMA.
:::

For more information about GPUDirect RDMA, see the following:

- [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- [Minimal GPUDirect RDMA Demonstration](https://github.com/NVIDIA/jetson-rdma-picoevb)
    source code, which provides a real hardware example of using RDMA
    and includes both kernel drivers and user space applications for
    the RHS Research PicoEVB and HiTech Global HTG-K800 FPGA boards.

(gxf-tech)=

## Graph Execution Framework

GXF (Graph Execution Framework) is an NVIDIA-internal graph execution framework that forms the foundation of the Holoscan SDK. GXF provides a low-level entity-component system for building and executing computation graphs, including schedulers, memory allocators, message passing, and a YAML-based graph definition format.

The Holoscan SDK provides a developer-friendly C++ and Python APIs that abstract away GXF internals, culminating in a fully native operator and application model. Today, most Holoscan SDK users do not need to interact with GXF directly.

### GXF core concepts

For historical context and to help interpret older code or documentation, here is a mapping of GXF concepts to their Holoscan SDK equivalents:

| GXF Concept | Holoscan SDK Equivalent | Description |
|---|---|---|
| **Entity** | (implicit) | A node in the computation graph; a container for components. In the Holoscan SDK, an Operator implicitly represents an entity. |
| **Codelet** | **{ref}`Operator <exhale_class_classholoscan_1_1Operator>`** | A component that executes custom code via lifecycle methods (`start`, `tick`/`compute`, `stop`). |
| **Component** | **{ref}`Resource <exhale_class_classholoscan_1_1Resource>`** | Supporting functionality such as memory allocators, clocks, or serializers attached to an entity. |
| **Scheduling Term** | **{ref}`Condition <exhale_class_classholoscan_1_1Condition>`** | A predicate that determines when an operator is ready for execution. |
| **Receiver / Transmitter** | **{ref}`Input / Output Port <exhale_class_classholoscan_1_1IOSpec>`** | Message-passing endpoints between operators. |
| **Connection** | **Flow (Edge)** | A directed edge in the application graph connecting an output port to an input port. |
| **Scheduler** | **{ref}`Scheduler <exhale_class_classholoscan_1_1Scheduler>`** | Orchestrates the execution of operators based on their conditions. |
| **GXF Extension** | **Operator / Resource library** | A shared library that registers components with the runtime. Native Holoscan operators do not require GXF extension registration. |

(tensorrt)=

## TensorRT Optimized Inference

[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) is a deep learning inference framework based on CUDA that provided the highest optimizations to run on NVIDIA GPUs, including the NVIDIA Developer Kits.

The {ref}`inference module<holoinfer>` leverages TensorRT among other backends, and provides the ability to execute multiple inferences in parallel.

(cuda_rendering_interop)=

## Interoperability between CUDA and rendering frameworks

Vulkan is commonly used for real-time visualization and, like CUDA, is executed on the GPU. This provides an opportunity for efficient sharing of resources between CUDA and this rendering framework.

The {ref}`Holoviz <visualization>` module uses the [external resource interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html) functions of the low-level CUDA driver application programming interface, the Vulkan [external memory](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory_fd.html) and [external semaphore](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_semaphore.html) extensions.

(npp)=

## Accelerated image transformations

Streaming image processing often requires common 2D operations like resizing, converting bit widths, and changing color formats. NVIDIA has built the CUDA accelerated NVIDIA Performance Primitive Library ([NPP](https://docs.nvidia.com/cuda/npp/index.html)) that can help with many of these common transformations. NPP is extensively showcased in the Format Converter operator of the Holoscan SDK.

(ucx)=

## Unified Communications X

The [Unified Communications X](https://openucx.org/) (UCX) framework is an open-source communication framework developed as a collaboration between industry and academia. It provides high-performance point-to-point communication for data-centric applications. Holoscan SDK uses UCX to send data between fragments in distributed applications. UCX's high level protocols attempt to automatically select an optimal transport layer depending on the hardware available. For example technologies such as [TCP](https://en.wikipedia.org/wiki/Transmission_Control_Protocol), CUDA memory copy, [CUDA IPC](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication) and [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) are supported.

(matx)=

## MatX Accelerated Computing

The Holoscan SDK integrates the [MatX](https://github.com/NVIDIA/MatX) library, a high-performance C++17 library for numerical computing on NVIDIA GPUs.

The library is accessible in C++ applications through the `matx::matx` interface library. It enables zero-copy data exchange between MatX tensors (`matx::tensor`) and `holoscan::Tensor` via the DLPack standard.

To use MatX in a C++ application, link against the `matx::matx` target in `CMakeLists.txt`:

```cmake
target_link_libraries(my_application
  PRIVATE
  holoscan::core
  matx::matx
)
```

A new C++ example, `matx_basic`, is available in the `examples/matx/matx_basic` directory to demonstrate creating, sharing, and performing GPU-accelerated operations on MatX tensors within a Holoscan pipeline.
