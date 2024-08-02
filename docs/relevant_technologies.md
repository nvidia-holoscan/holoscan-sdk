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

(gpudirect_rdma)=
## Rivermax and GPUDirect RDMA

The NVIDIA Developer Kits equipped with a [ConnectX network adapter](https://www.nvidia.com/en-us/networking/ethernet-adapters/) can be used along with the [NVIDIA Rivermax SDK](https://developer.nvidia.com/networking/rivermax) to provide an extremely efficient network connection that is further optimized for GPU workloads by using [GPUDirect](https://developer.nvidia.com/gpudirect) for RDMA. This technology avoids unnecessary memory copies and CPU overhead by copying data directly to or from pinned GPU memory, and supports both the integrated GPU or the discrete GPU.

:::{note}
NVIDIA is also committed to supporting hardware vendors enable RDMA within their own drivers, an example of which is provided by the {ref}`aja_video_systems` as part of a partnership with
NVIDIA for the Holoscan SDK. The `AJASource` operator is an example of how the SDK can leverage RDMA.
:::

For more information about GPUDirect RDMA, see the following:

- [GPUDirect RDMA Documentation](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html)
- [Minimal GPUDirect RDMA Demonstration](https://github.com/NVIDIA/jetson-rdma-picoevb)
    source code, which provides a real hardware example of using RDMA
    and includes both kernel drivers and userspace applications for
    the RHS Research PicoEVB and HiTech Global HTG-K800 FPGA boards.

(gxf-tech)=
## Graph Execution Framework

The Graph Execution Framework (GXF) is a core component of the Holoscan SDK that provides features to execute pipelines of various independent tasks with high performance by minimizing or removing the need to copy data across each block of work, and providing ways to optimize memory allocation.

GXF will be mentioned in many places across this user guide, including a {ref}`dedicated section <gxf-user-guide>` which provides more details.

(tensorrt)=
## TensorRT Optimized Inference

[NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) is a deep learning inference framework based on CUDA that provided the highest optimizations to run on NVIDIA GPUs, including the NVIDIA Developer Kits.

The {ref}`inference module<holoinfer>` leverages TensorRT among other backends, and provides the ability to execute multiple inferences in parallel.

(cuda_rendering_interop)=
## Interoperability between CUDA and rendering frameworks

Vulkan is commonly used for realtime visualization and, like CUDA, is executed on the GPU. This provides an opportunity for efficient sharing of resources between CUDA and this rendering framework.

The {ref}`Holoviz <visualization>` module uses the [external resource interoperability](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXTRES__INTEROP.html) functions of the low-level CUDA driver application programming interface, the Vulkan [external memory](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_memory_fd.html) and [external semaphore](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VK_KHR_external_semaphore.html) extensions.

(npp)=
## Accelerated Image Transformations

Streaming image processing often requires common 2D operations like resizing, converting bit widths, and changing color formats. NVIDIA has built the CUDA accelerated NVIDIA Performance Primitive Library ([NPP](https://docs.nvidia.com/cuda/npp/index.html)) that can help with many of these common transformations. NPP is extensively showcased in the Format Converter operator of the Holoscan SDK.

(ucx)=
## Unified Communications X

The [Unified Communications X](https://openucx.org/) (UCX) framework is an open-source communication framework developed as a collaboration between industry and academia. It provides high performance point-to-point communication for data-centric applications. Holoscan SDK uses UCX to send data between fragments in distributed applications. UCX's high level protocols attempt to automatically select an optimal transport layer depending on the hardware available. For example technologies such as [TCP](https://en.wikipedia.org/wiki/Transmission_Control_Protocol), CUDA memory copy, [CUDA IPC](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#interprocess-communication) and [GPUDirect RDMA](https://docs.nvidia.com/cuda/gpudirect-rdma/index.html) are supported.
