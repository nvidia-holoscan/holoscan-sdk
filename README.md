# Holoscan SDK

The **Holoscan SDK** is part of [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

> In previous releases, the prefix [`Clara`](https://developer.nvidia.com/industries/healthcare) was used to define Holoscan as a platform designed initially for [medical devices](https://www.nvidia.com/en-us/clara/developer-kits/). As Holoscan has grown, its potential to serve other areas has become apparent. With version 0.4.0, we're proud to announce that the Holoscan SDK is now officially built to be domain-agnostic and can be used to build sensor AI applications in multiple domains. Note that some of the content of the SDK (sample applications) or the documentation might still appear to be healthcare-specific pending additional updates. Going forward, domain specific content will be hosted on the [HoloHub](https://nvidia-holoscan.github.io/holohub) repository.

Visit the [NGC demo website](https://demos.ngc.nvidia.com/holoscan) for a live demonstration of some of Holoscan capabilities.

## Table of Contents

- [Getting Started](#getting-started)
- [Building the SDK from source](#building-the-sdk-from-source)
  - [Prerequisites](#prerequisites)
  - [(Recommended) using the `run` script](#recommended-using-the-run-script)
  - [Cross-compilation](#cross-compilation)
  - [(Advanced) Docker + CMake](#advanced-docker-cmake)
  - [(Advanced) Local environment + CMake](#advanced-local-environment-cmake)
- [Utilities](#utilities)
  - [Code coverage](#code-coverage)
  - [Linting](#linting)
  - [VSCode](#working-with-visual-studio-code)
- [Troubleshooting](#troubleshooting)
- [Repository structure](#repository-structure)

## Getting Started

Visit the Holoscan User Guide to get started with the Holoscan SDK: <https://docs.nvidia.com/holoscan/sdk-user-guide/getting_started.html>

## Building the SDK from source

> **⚠️ Disclaimer**: we only recommend building the SDK from source if you are a developer of the SDK, or need to build the SDK with debug symbols or other options not used as part of the published packages. If you want to write your own operator or application, you can use the SDK as a dependency (and contribute to [HoloHub](https://github.com/nvidia-holoscan/holohub)). If you need to make other modifications to the SDK, [file a feature or bug request](https://forums.developer.nvidia.com/c/healthcare/holoscan-sdk/320/all). If that's not the case, prefer installing the SDK from [published packages](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#install-the-sdk).

### Prerequisites

- Prerequisites for each supported platform are documented in [the user guide](https://docs.nvidia.com/holoscan/sdk-user-guide/sdk_installation.html#prerequisites).
- Additionally, to build and run the SDK in a containerized environment (recommended) you'll need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) with Docker.

### (Recommended) Using the `run` script

Call **`./run build`** within the repository to build the build container and the CMake project.

- *If you encounter errors during the CMake build, you can execute `./run clear_cache` to remove cache/build/install folders*
- *Execute `./run build --help` for more information*
- *Execute `./run build --dryrun` to see the commands that will be executed*
- *That command can be broken-up in more granular commands also:*

  ```sh
  ./run setup # setup Docker
  ./run build_image # create the build Docker container
  ./run build # run the CMake configuration, build, and install steps
  ```

Call the **`./run launch`** command to start and enter the build container.

- *You can run from the `install` or `build` tree by passing the working directory as an argument (ex: `./run launch install`)*
- *Execute `./run launch --help` for more information*
- *Execute `./run launch --dryrun` to see the commands that will be executed*

Run the [**examples**](./examples#readme) inside the container by running their respective commands listed within each directory README file.

### Cross-compilation

While the Dockerfile to build the SDK does not currently support true cross-compilation, you can compile the Holoscan SDK for the developer kits (arm64) from a x86_64 host using an emulation environment.

1. [Install qemu](https://github.com/multiarch/qemu-user-static)
2. Clear your build cache: `./run clear_cache`
3. Rebuild for `linux/arm64` using `--platform` or `HOLOSCAN_BUILD_PLATFORM`:
    - `./run build --platform linux/arm64`
    - `HOLOSCAN_BUILD_PLATFORM=linux/arm64 ./run build`

You can then copy the `install` folder generated by CMake to a developer kit with a configured environment or within a container to use for running and developing applications.

### (Advanced) Docker + CMake

The [`run`](./run) script mentioned above is helpful to understand how Docker and CMake are configured and run, as commands will be printed when running it or using `--dryrun`.
We recommend looking at those commands if you want to use Docker and CMake manually, and reading the comments inside the script for details about each parameter (specifically the `build()` and `launch()` methods).

### (Advanced) Local environment + CMake

> **⚠️ Disclaimer**: this method of building the SDK is not actively tested or maintained. Instructions below might go out of date.

To build the Holoscan SDK on a local environment, the following dev dependencies are needed. The last column refers to the stage (`FROM`) in the [Dockerfile](./Dockerfile) where respective commands can be found to build/install these dependencies.

| Dependency | Min version | Needed by | Dockerfile stage |
|---|---|---|---|
| CUDA | 11.4.0 | Core SDK | base |
| gRPC | 1.54.2 | Core SDK | grpc-builder |
| UCX | 1.14.0 | Core SDK | ucx-builder |
| GXF | 23.05 | Core SDK | gxf-downloader |
| TensorRT | 8.2.3 | Inference operator | base |
| ONNX Runtime | 1.12.1 | Inference operator | onnxruntime-downloader |
| LibTorch | 1.12.0 | Inference operator<br>(torch plugin) | torch-downloader-[x86_64\|arm64] |
| TorchVision | 0.14.1 | Inference operator<br>(torch plugin) | torchvision-builder-[x86_64\|arm64] |
| Vulkan SDK | 1.3.216 | Holoviz operator | vulkansdk-builder |
| Vulkan loader and<br>validation layers | 1.2.131 | Holoviz operator | dev |
| spirv-tools | 2020.1 | Holoviz operator | dev |
| V4L2 | 1.18.0 | V4L2 operator | dev |
| CMake | 3.22.2 | Build process | build-tools |
| Patchelf | N/A | Build process | build-tools |

Note: refer to the [Dockerfile](./Dockerfile) for other dependencies which are not needed to build, but might be needed for:

- runtime (openblas for torch, egl for headless rendering, cloudpickle for distributed python apps, cupy for some examples...)
- testing (lcov, valgrind, pytest...)
- utilities (ffmpeg, v4l-utils, ...)

For CMake to find these dependencies, install them in default system paths, or pass `CMAKE_PREFIX_PATH`, `CMAKE_LIBRARY_PATH`, and/or `CMAKE_INCLUDE_PATH` during configuration.

##### Example

```sh
# Configure
cmake -S $source_dir -B $build_dir \
  -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CUDAToolkit_ROOT:PATH="/usr/local/cuda"

# Build
cmake --build $build_dir -j

# Install
cmake --install $build_dir --prefix $install_dir
```

The commands to run the [**examples**](./examples#readme) are then the same as in the dockerized environment, and can be found in the respective source directory READMEs.

## Utilities

Some utilities are available in the [`scripts`](./scripts) folder, others closer to the built process are listed below:

### Code coverage

To generate a code coverage report use the following command after completing setup:

```sh
./run coverage
```

Open the file `build/coverage/index.html` to access the interactive coverage web tool.

### Linting

Run the following command to run various linting tools on the repository:

```sh
./run lint # optional: specify directories
```

> Note: Run `run lint --help` to see the list of tools that are used. If a lint command fails due to a missing module or executable on your system, you can install it using `python3 -m pip install <tool>`.

### VSCode

Visual Studio Code can be utilized to develop the Holoscan SDK. The `.devcontainer` folder holds the configuration for setting up a [development container](https://code.visualstudio.com/docs/remote/containers) with all necessary tools and libraries installed.

The `./run` script contains `vscode` and `vscode_remote` commands for launching Visual Studio Code in a container or from a remote machine, respectively.

- To launch Visual Studio Code in a dev container, use `./run vscode`.
- To attach to an existing dev container from a remote machine, use `./run vscode_remote`. For more information, refer to the instructions from `./run vscode_remote -h`.

Once Visual Studio Code is launched, the development container will be built and the recommended extensions will be installed automatically, along with CMake being configured.

#### Configuring CMake in the Development Container

For manual configuration of CMake, open the command palette (`Ctrl + Shift + P`) and run the `CMake: Configure` command.

#### Building the Source Code in the Development Container

The source code in the development container can be built by either pressing `Ctrl + Shift + B` or executing `Tasks: Run Build Task` from the command palette (`Ctrl + Shift + P`).

#### Debugging the Source Code in the Development Container

To debug the source code in the development container, open the `Run and Debug` view (`Ctrl + Shift + D`), select a debug configuration from the dropdown list, and press `F5` to initiate debugging.

## Troubleshooting

### X11: Failed to open display :0 [...] Failed to initialize GLFW

Enable permissions to your X server from Docker, either:

- Passing `-u $(id -u):$(id -g)` to `docker run`, or
- Running `xhost +local:docker` on your host

### GLX: Failed to create context: GLXBadFBConfig

You may encounter the error message if the Holoscan Application runs on a Virtual Machine (by a Cloud Service Provider) or without a physical display attached. If you want to run OpenGL applications that use GPU on x11 (e.g., VNC or NoMachine), the following environment variables need to be set before executing the application to offload the rendering to GPU.

```sh
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

### `GXF_ENTITY_COMPONENT_NOT_FOUND` or `GXF_ENTITY_NOT_FOUND`

Ensure all your application connections in the yaml file (`nvidia::gxf::Connection`) refer to entities or components defined within. This can occur when attempting to remove a component and not cleaning up the stale connections.

### No receiver connected to transmitter of <scheduling term id> of entity <x>. The entity will never tick

Ensure your entity or component is not an orphan, but is connected to a `nvidia::gxf::Connection`.

### AJA device errors

These errors indicate that you don't have AJA support in your environment.

```sh
2022-06-09 18:45:13.826 ERROR gxf_extensions/aja/aja_source.cpp@80: Device 0 not found.
2022-06-09 18:45:13.826 ERROR gxf_extensions/aja/aja_source.cpp@251: Failed to open device 0
```

Double check that you have installed the AJA ntv2 driver, loaded the driver after every reboot, and that you have specified `--device /dev/ajantv20:/dev/ajantv20` in the `docker run` command if you’re running a docker container.

### GXF format converter errors

These errors may indicate that you need to reconfigure your format converter's num_block number.

```sh
2022-06-09 18:47:30.769 ERROR gxf_extensions/format_converter/format_converter.cpp@521: Failed to allocate memory for the channel conversion
2022-06-09 18:47:30.769 ERROR gxf_extensions/format_converter/format_converter.cpp@359: Failed to convert tensor format (conversion type:6)
```

Try increasing the current num_block number by 1 in the yaml file for all format converter entities. This may happen if your yaml file was configured for running with RDMA and you have decided to disable RDMA.

### Video device error

Some of those errors may occur when running the V4L2 codelet:

```
Failed to open device, OPEN: No such file or directory
```

Ensure you have a video device connected (ex: USB webcam) and listed when running `ls -l /dev/video*`.

```
Failed to open device, OPEN: Permission denied
```

This means the `/dev/video*` device is not available to the user from within docker. Give `--group-add video` to the `docker run` command.

### HolovizOp fails on hybrid GPU systems with non-NVIDIA integrated GPU and NVIDIA discrete GPU

You may encounter an error when trying to run the Holoviz operator on a laptop equipped with an integrated and a discrete GPU. By default these systems will be using the integrated GPU when running an application. The integrated GPU does not provide the capabilities the Holoviz operator needs and the operator will fail.

The following environment variables need to be set before executing the application to offload the rendering to the discrete GPU. See [PRIME Render Offload](https://download.nvidia.com/XFree86/Linux-x86_64/535.54.03/README/primerenderoffload.html) for more information.

```sh
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

## Repository structure

The repository is organized as such:

- `cmake/`: CMake configuration files
- `data/`: directory where data will be downloaded
- `examples/`: source code for the examples
- `gxf_extensions/`: source code for the holoscan SDK gxf codelets
- `include/`: source code for the holoscan SDK core
- `modules/`: source code for the holoscan SDK modules
- `patches/`: patch files applied to dependencies
- `python/`: python bindings for the holoscan SDK
- `scripts/`: utility scripts
- `src/`: source code for the holoscan SDK core
- `tests/`: tests for the holoscan SDK
