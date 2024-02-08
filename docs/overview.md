# Overview

[NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk) is the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

:::{note}
In previous releases, the prefix [`Clara`](https://developer.nvidia.com/industries/healthcare) was used to define Holoscan as a platform designed initially for [medical devices](https://www.nvidia.com/en-us/clara/developer-kits/). As Holoscan has grown, its potential to serve other areas has become apparent. With version 0.4.0, we're proud to announce that the Holoscan SDK is now officially built to be domain-agnostic and can be used to build sensor AI applications in multiple domains. Note that some of the content of the SDK (sample applications) or the documentation might still appear to be healthcare-specific pending additional updates. Going forward, domain specific content will be hosted on the [HoloHub](https://nvidia-holoscan.github.io/holohub) repository.
:::

The Holoscan SDK assists developers by providing:

1. **Various installation strategies**

From containers, to python wheels, to source, from development to deployment environments, the Holoscan SDK comes in many packaging flavors to adapt to different needs. Find more information in the {ref}`sdk installation<clara_holoscan_setup>` section.

2. **C++ and Python APIs**

These APIs are now the recommended interface for the creation of application pipelines in the Holoscan SDK. See the {ref}`Using the SDK <holoscan-user-overview>` section to learn how to leverage those APIs, or the Doxygen pages ([C++](api/holoscan_cpp_api.md)/[Python](api/holoscan_python_api.md)) for specific API documentation.

3. **Built-in Operators**

The units of work of Holoscan applications are implemented within Operators, as described in the [core concepts](holoscan_core.md) of the SDK. The operators included in the SDK provide domain-agnostic functionalities such as IO, machine learning inference, processing, and visualization, optimized for AI streaming pipelines, relying on a set of [Core Technologies](relevant_technologies.md). This guide provides more information on the operators provided within the SDK [here](holoscan_operators_extensions.md).

4. **Minimal Examples**

The Holoscan SDK provides a list of examples to illustrate specific capabilities of the SDK. Their source code can be found in the [GitHub repository](https://github.com/nvidia-holoscan/holoscan-sdk/tree/main/examples#readme). The {ref}`Holoscan by Example<holoscan-getting-started>` section provides step-by-step analysis of some of these examples to illustrate the innerworkings of the Holoscan SDK.

5. **Repository of Operators and Applications**

[HoloHub](https://nvidia-holoscan.github.io/holohub) is a central repository for users and developers to share reusable operators and sample applications with the Holoscan community. Being open-source, these operators and applications can also be used as reference implementations to complete the built-in operators and examples available in the SDK.

6. **Tooling to Package and Deploy Applications**

Packaging and deploying applications is a complex problem that can require large amount of efforts. The [Holoscan CLI](./cli/cli.md) is a command-line interface included in the Holoscan SDK that provides commands to [package and run applications](./holoscan_packager.md) in OCI-compliant containers that could be used for production.

7. **Performance tools**

As highlighted in the relevant technologies section, the soul of the Holoscan project is to achieve peak performance by leveraging hardware and software developed at NVIDIA or provided by third-parties. To validate this, Holoscan provides performance tools to help users and developers track their application performance. They currently include:

- a {ref}`Video Pipeline Latency Measurement Tool <latency_tool>` to measure and estimate the total end-to-end latency of a video streaming application including the video capture, processing, and output using various hardware and software components that are supported by the Holoscan Developer Kits.
- the [Data Flow Tracking](./flow_tracking.md) feature to profile your application and analyze the data flow between operators in its graph.

8. **Container to leverage both iGPU and dGPU on Holoscan devkits**

The Holoscan developer kits can - at this time - only be flashed to leverage the integrated GPU (Tegra SoC) or the added discrete GPU. The [L4T Compute Assist container](./use_igpu_with_dgpu.md) on NGC is a mechanism to leverage both concurrently.

9. **Documentation**

The Holoscan SDK documentation is composed of:

- This user guide, in a [webpage](https://docs.nvidia.com/holoscan/sdk-user-guide/) or [PDF](https://developer.nvidia.com/downloads/holoscan-sdk-user-guide) format
- Build and run instructions specific to each {ref}`installation strategy<clara_holoscan_setup>`
- [Release notes](https://github.com/nvidia-holoscan/holoscan-sdk/releases) on Github
