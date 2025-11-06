# Getting Started with Holoscan

As described in the [Overview](./overview.md), the Holoscan SDK provides many components and capabilities. The goal of this section is to provide a recommended path to getting started with the SDK.

## 1. Choose your platform

The Holoscan SDK is optimized and compatible with multiple hardware platforms, including NVIDIA Developer Kits (aarch64), DGX Spark, and x86_64 workstations. Learn more on the [developer page](https://developer.nvidia.com/holoscan-sdk) to help you decide what hardware you should target.

## 2. Setup the SDK and your platform

Start with [installing the SDK](./sdk_installation.md). If you have a need for it, you can go through additional [recommended setups](./additional_setup.md) to achieve peak performance, or [setup additional sensors](./third_party_hw_setup.md) from NVIDIA's partners.

## 3. Learn the framework

1. Start with the [Core Concepts](./holoscan_core.md) to understand the technical terms used in this guide, and the overall behavior of the framework.
2. Learn how to use the SDK in one of two ways (or both), based on your preference:
   a. Going through the [Holoscan by Example](./holoscan_by_example.md) tutorial which will build your knowledge step-by-step by going over concrete minimal examples in the SDK. You can refer to each example source code and run instructions to inspect them and run them as you go.
   b. Going through the condensed documentation that should cover all capabilities of the SDK using minimal mock code snippets, including [creating an application](./holoscan_create_app.md), [creating a distributed application](./holoscan_create_distributed_app.md), and [creating operators](./holoscan_create_operator.md).

## 4. Understand the reusable capabilities of the SDK

The Holoscan SDK does not only provide a framework to build and run applications, but also a set of reusable operators to facilitate implementing applications for streaming, AI, and other general domains.

The list of existing operators is available [here](./holoscan_operators_extensions.md), which points to the C++ or Python API documentation for more details. Specific documentation is available for the [visualization](./visualization.md) (codename: HoloViz) and [inference](./inference.md) (codename: HoloInfer) operators.

Additionally, [HoloHub](https://github.com/nvidia-holoscan/holohub) is a central repository for users and developers to share reusable operators and sample applications with the Holoscan community, extending the capabilities of the SDK:

- Just like the SDK operators, the HoloHub operators can be used in your own Holoscan applications.
- The HoloHub sample applications can be used as reference implementations to complete the examples available in the SDK.

Take a glance at [HoloHub](https://github.com/nvidia-holoscan/holohub) to find components you might want to leverage in your application, improve upon existing work, or contribute your own additions to the Holoscan platform.

## 5. Write and run your own application

The steps above cover what is required to write your own application and run it. For facilitating packaging and distributing, the Holoscan SDK includes utilities to [package and run your Holoscan application](./holoscan_packager.md) in an OCI-compliant container image.

## 6. Master the details

- Expand your understanding of the framework with details on the [logging utility](./holoscan_logging.md) or the [data flow tracking](./flow_tracking.md) benchmarking tool and [job statistics](./gxf_job_statistics) measurements.
- Learn more details on the configurable components that control the execution of your application, like [Schedulers], [Conditions], and [Resources]. (Advanced) These components are part on the GXF execution backend, hence the **Graph Execution Framework** section at the bottom of this guide if deep understanding of the application execution is needed.
