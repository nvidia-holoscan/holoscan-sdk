# Holoscan SDK

The **Holoscan SDK** is part of [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

## Table of Contents

- [Getting Started](#getting-started)
- [Obtaining the Holoscan SDK](#obtaining-the-holoscan-sdk)
- [AI Coding Assistants](#ai-coding-assistants)
- [Troubleshooting and Feedback](#troubleshooting-and-feedback)
- [Contributing to Holoscan SDK](#contributing-to-holoscan-sdk)
- [Repository structure](#repository-structure)

## Getting Started

Visit the Holoscan User Guide to get started with the Holoscan SDK: <https://docs.nvidia.com/holoscan/sdk-user-guide/getting_started.html>

The Holoscan User Guide includes:

- An introduction to the NVIDIA Holoscan platform, including the Holoscan C++/Python SDK;
- Requirements and setup steps;
- Detailed SDK documentation, including a developer introduction, examples, and API details.

We also recommend visiting [NVIDIA HoloHub](https://github.com/nvidia-holoscan/holohub) to view
community projects and reusable components available for your Holoscan project.

## Obtaining the Holoscan SDK

The Holoscan User Guide documents several options to install and run the Holoscan SDK:

- As an [NGC Container 🐋](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#sd-tab-item-2)
- As a [Debian Package 📦️](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#sd-tab-item-3)
- As a [Python Wheel 🐍](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#sd-tab-item-4)

Visit the [Holoscan User Guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#not-sure-what-to-choose) for
guidance to help choose which installation option may be right for your use case.

If the options above do not support your use case, you may prefer to [build the SDK from source](./DEVELOP.md).

Please review [Holoscan SDK prerequisites](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#prerequisites)
before getting started.

## AI Coding Assistants

The Holoscan SDK 5.x user guide includes
[AI Coding Assistants](https://docs.nvidia.com/holoscan/sdk-user-guide/resources/ai-coding-agents)
guidance for grounding AI coding assistants in the 5.x API, examples, and
compile-time diagnostics. A dedicated Holoscan coding-assistant skill package
is planned; until then, use the guide page as the source of truth for
assistant-facing 5.x development guidance.

## Troubleshooting and Feedback

We appreciate community discussion and feedback in support of Holoscan platform users and developers. We ask that users:

- Review the [Holoscan SDK Frequently Asked Questions](docs/hsdk_faq.mdx) document for common solutions and workarounds.
- Direct questions to the [NVIDIA Support Forum](https://forums.developer.nvidia.com/c/robotics-edge-computing/holoscan/757).
- Enter SDK issues on the [SDK GitHub Issues board](https://github.com/nvidia-holoscan/holoscan-sdk/issues).

## Contributing to Holoscan SDK

Holoscan SDK is developed internally and released as open source software. We welcome community contributions
and may include them in Holoscan SDK releases at our discretion. Please refer to the Holoscan SDK
[Contributing Guide](/CONTRIBUTING.md) for more information.

## Repository structure

The repository is organized as such:

- `cmake/`: CMake configuration files
- `data/`: created during the build when dataset download is enabled
- `examples/`: source code for the examples
- `include/`: source code for the holoscan SDK core
- `modules/`: source code for optional SDK modules, including Inference, Vision, and Visualization
- `patches/`: patch files applied to dependencies
- `python/`: python bindings for the holoscan SDK
- `scripts/`: utility scripts
- `src/`: source code for the holoscan SDK core
- `tests/`: tests for the holoscan SDK
