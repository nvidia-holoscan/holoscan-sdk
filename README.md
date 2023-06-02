# Holoscan SDK

The **Holoscan SDK** is part of [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk), the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

> In previous releases, the prefix [`Clara`](https://developer.nvidia.com/industries/healthcare) was used to define Holoscan as a platform designed initially for [medical devices](https://www.nvidia.com/en-us/clara/developer-kits/). As Holoscan has grown, its potential to serve other areas has become apparent. With version 0.4.0, we're proud to announce that the Holoscan SDK is now officially built to be domain-agnostic and can be used to build sensor AI applications in multiple domains. Note that some of the content of the SDK (sample applications) or the documentation might still appear to be healthcare-specific pending additional updates. Going forward, domain specific content will be hosted on the [HoloHub](https://github.com/nvidia-holoscan/holohub) repository.

Visit the [NGC demo website](https://demos.ngc.nvidia.com/holoscan) for a live demonstration of some of Holoscan capabilities.

## Table of Contents
- [Documentation](#documentation)
- [Using the released SDK](#using-the-released-sdk)
- [Building the SDK from source](#building-the-sdk-from-source)
  - [Prerequisites](#prerequisites)
    - [For Holoscan Developer Kits (aarch64)](#for-holoscan-developer-kits-aarch64)
    - [For x86_64 systems](#for-x86_64-systems)
  - [Recommended: using the `run` script](#recommended-using-the-run-script)
  - [Cross-compilation](#cross-compilation)
  - [Advanced: Docker + CMake](#advanced-docker-cmake)
  - [Advanced: Local environment + CMake](#advanced-local-environment-cmake)
- [Utilities](#utilities)
  - [Code coverage](#code-coverage)
  - [Linting](#linting)
- [Troubleshooting](#troubleshooting)
- [Repository structure](#repository-structure)

## Documentation

* The latest SDK user guide is available at https://docs.nvidia.com/clara-holoscan.
* For a full list of Holoscan documentation, visit the [Holoscan developer page](https://developer.nvidia.com/clara-holoscan-sdk).

## Using the released SDK

The Holoscan SDK is available as part of the following packages:
- 🐋 The [Holoscan container image on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) includes the Holoscan libraries, GXF extensions, headers, example source code, and sample datasets, as well as all the dependencies that were tested with Holoscan. It is the recommended way to run the Holoscan examples, while still allowing you to create your own C++ and Python Holoscan application.
- 🐍 The [Holoscan python wheels on PyPI](https://pypi.org/project/holoscan/) are the simplest way for Python developers to get started with the SDK using `pip install holoscan`. The wheels include the SDK libraries, not the example applications or the sample datasets.
- 📦️ The [Holoscan Debian package on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_dev_deb) includes the libraries, headers, example applications and CMake configurations needed for both C++ and Python developers. It does not include sample datasets.

## Building the SDK from source

> **Disclaimer**: we only recommend building the SDK from source if you are a developer of the SDK, or need to build the SDK with debug symbols or other options not used as part of the released packages. If you want to modify some code to fit your use case, contribute to HoloHub if it's an operator or application, or file a feature or bug request. If that's not the case, prefer using [generated packages](#using-the-released-sdk).

### Prerequisites

The Holoscan SDK currently supports the [Holoscan Developer Kits](https://www.nvidia.com/en-us/clara/developer-kits) (aarch64) as well as x86_64 systems.

#### For Holoscan Developer Kits (aarch64)

Set up your developer kit:

Developer Kit | User Guide | HoloPack | GPU
------------- | ---------- | -------- | ---
[NVIDIA IGX Orin](https://www.nvidia.com/en-us/edge-computing/products/igx/) | [Guide](https://github.com/nvidia-holoscan/holoscan-docs/blob/main/devkits/nvidia-igx-orin/nvidia_igx_orin_user_guide.md) | 2.0 | dGPU or iGPU
NVIDIA IGX Orin [ES] | [Guide](https://developer.download.nvidia.com/CLARA/IGX-Orin-Developer-Kit-User-Guide-(v1.0).pdf) | 1.2 | dGPU only
[NVIDIA Clara AGX](https://www.nvidia.com/en-gb/clara/intelligent-medical-instruments/) | [Guide](https://developer.nvidia.com/clara-agx-developer-kit-user-guide) | 1.2 | dGPU only

> Notes:
>
> - For Rivermax support (optional/local development only at this time), GPUDirect drivers need to be loaded manually at this time, see the [User Guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/additional_setup.html#setting-up-gpudirect-rdma).
> - Refer to the user guide for additional steps needed to support third-party technologies, such as [AJA cards](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html).
> - Additional dependencies are required when developing locally instead of using a containerized environment, see details in the [section below](#advanced-local-environment--cmake).

#### For x86_64 systems

You'll need the following to build applications from source on x86_64:
- OS: Ubuntu 20.04
- NVIDIA discrete GPU (dGPU)
  - Ampere or above recommended for best performance
  - [Quadro/NVIDIA RTX](https://www.nvidia.com/en-gb/design-visualization/desktop-graphics/) necessary for [GPUDirect RDMA](https://developer.nvidia.com/gpudirect) support
  - Tested with [NVIDIA RTX 6000](https://www.nvidia.com/en-us/design-visualization/rtx-6000/) and [NVIDIA RTX A6000](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)
- [NVIDIA dGPU drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes): 510.73.08 or above
- For containerized development (recommended):
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- For local development (advanced):
  - See details in the [section below](#advanced-local-environment--cmake).
- For Rivermax support (optional/local development only at this time):
  - [NVIDIA ConnectX SmartNIC](https://www.nvidia.com/en-us/networking/ethernet-adapters/)
  - [OFED Network Drivers](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/): 5.8
  - [GPUDirect Drivers](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/additional_setup.html#setting-up-gpudirect-rdma): 1.1
  - [Rivermax SDK](https://developer.nvidia.com/networking/rivermax): 1.20

### Recommended: using the `run` script

Call **`./run build`** within the repository to build the development container and the CMake project.
* *If you encounter errors during the CMake build, you can execute `./run clear_cache` to remove cache/build/install folders*
* *Execute `./run build --help` for more information*
* *Execute `./run build --dryrun` to see the commands that will be executed*
* *That command can be broken-up in more granular commands also:*
  ```sh
  ./run setup # setup Docker
  ./run build_image # create the development Docker container
  ./run install_gxf # install the GXF package (default location is `.cache/gxf`)
  ./run build # run the CMake configuration, build, and install steps
  ```

Call the **`./run launch`** command to start and enter the development container.
* *You can run from the `install` or `build` tree by passing the working directory as an argument (ex: `./run launch install`)*
* *Execute `./run launch --help` for more information*
* *Execute `./run launch --dryrun` to see the commands that will be executed*

Run the [**examples**](./examples) inside the container by running their respective commands listed within each directory README file.

### Cross-compilation

While the development Dockerfile does not currently support true cross-compilation, you can compile the Holoscan SDK for the developer kits (arm64) from a x86_64 host using an emulation environment.

1. [Install qemu](https://docs.nvidia.com/datacenter/cloud-native/playground/x-arch.html#emulation-environment)
2. Clear your build cache: `./run clear_cache`
3. Rebuild for `linux/arm64` using `--platform` or `HOLOSCAN_BUILD_PLATFORM`:
    * `./run build --platform linux/arm64`
    * `HOLOSCAN_BUILD_PLATFORM=linux/arm64 ./run build`

You can then copy the `install` folder generated by CMake to a developer kit with a configured environment or within a container to use for running and developing applications.

### Advanced: Docker + CMake

The [`run`](./run) script mentioned above is helpful to understand how Docker and CMake are configured and run, as commands will be printed when running it or using `--dryrun`.
We recommend looking at those commands if you want to use Docker and CMake manually, and reading the comments inside the script for details about each parameter (specifically the `build()` and `launch()` methods).

### Advanced: Local environment + CMake

> **Disclaimer**: the risk of this section getting out of date is high since this is not actively maintained and tested at this time. Look at the dependencies pulled in the `Dockerfile` and the CMake commands in the `run` script if you run into issues.

To build on your local environment, you'll need to install the dependencies below in addition to the ones listed in the [prerequisites](#prerequisites):
- **CUDA Toolkit**: 11.6.1
  - `arm64`: already installed by SDK Manager on the Developer Kits
  - `x86_64`: [follow official instructions](https://developer.nvidia.com/cuda-11-6-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04)
- **TensorRT** 8.2.3
  - `arm64`: already installed by SDK Manager on the Developer Kits
  - `x86_64`: [follow official instructions](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-823/install-guide/index.html)
- **python-dev**: 3.8.10
  - `sudo apt install python3.8-dev`
- apt dependencies (**CMake**, **patchelf**, **OpenGL**, **Xorg** stack...)
  - see the `base` stage in [`Dockerfile`](Dockerfile)
  - see the `dev` stage in [`Dockerfile`](Dockerfile)
- **ONNX Runtime**: 1.12.1
  - download: see the `onnxruntime-downloader` stage in [`Dockerfile`](Dockerfile)
  - install: see the `dev` stage in [`Dockerfile`](Dockerfile)
- **Vulkan SDK**: 1.3.216
  - download & build: see the `vulkansdk-builder` stage in [`Dockerfile`](Dockerfile)
  - install: see the `dev` stage in [`Dockerfile`](Dockerfile)
- **GXF**: 2.5
  - call `./run install_gxf` or look at its content

You'll then need CMake to find those dependencies during configuration. Use `CMAKE_PREFIX_PATH`, `CMAKE_LIBRARY_PATH`, and/or `CMAKE_INCLUDE_PATH` if they're not installed in default system paths and cannot automatically be found.

##### Example:
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

The commands to run the [**examples**](./examples) are then the same as in the dockerized environment, and can be found in the respective source directory READMEs.

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

### Working with Visual Studio Code

Visual Studio Code can be utilized to develop Holoscan SDK. The `.devcontainer` folder holds the configuration for setting up a [development container](https://code.visualstudio.com/docs/remote/containers) with all necessary tools and libraries installed.

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

### No receiver connected to transmitter of <scheduling term id> of entity <x>. The entity will never tick.

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
This means the `/dev/video*` device is not available to the user from within docker. You can make it publicly available with `sudo chmod 666 /dev/video*` from your host.

### Cuda driver error 101 (CUDA_ERROR_INVALID_DEVICE): invalid device ordinal

The error can happen when running in a multi-GPU environment:

```
[2023-02-10 10:42:31.039] [holoscan] [error] [gxf_wrapper.cpp:68] Exception occurred for operator: 'holoviz' - Cuda driver error 101 (CUDA_ERROR_INVALID_DEVICE): invalid device ordinal
2023-02-10 10:42:31.039 ERROR gxf/std/entity_executor.cpp@509: Failed to tick codelet holoviz in entity: holoviz code: GXF_FAILURE
```

This is due to the fact that operators in your application run on different GPUs, depending on their internal implementation, preventing data from being freely exchanged between them.

To fix the issue, either:
1. Configure or modify your operators to copy data across the appropriate GPUs (using [`cuPointerGetAttribute`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__UNIFIED.html#group__CUDA__UNIFIED_1g0c28ed0aff848042bc0533110e45820c) and [`cuMemCpyPeerAsync()`](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g82fcecb38018e64b98616a8ac30112f2) internally). This comes at a cost and should only be done when leveraging multiple GPUs is required for improving performance (parallel processing).
2. Configure or modify your operators to use the same GPU (using [`cudaSetDevice()`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g159587909ffa0791bbe4b40187a4c6bb) internally).
3. Restrict your environment using [`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=CUDA_VISIBLE_DEVICES#cuda-environment-variables) to only expose a single GPU:
   ```sh
   export CUDA_VISIBLE_DEVICES=<GPU_ID>
   ```

Note that this generally occurs because the `HolovizOp` operator needs to use the GPU connected to the display, but other operators in the Holoscan pipeline might default to another GPU depending on their internal implementation. The index of the GPU connected to the display can be found by running `nvidia-smi` from a command prompt and looking for the GPU where the `Disp.A` value shows `On`. In the example below, the GPU `0` should be used before passing data to `HolovizOp`.

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A5000    On   | 00000000:09:00.0  On |                  Off |
| 30%   29C    P8    11W / 230W |    236MiB / 24564MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA RTX A5000    On   | 00000000:10:00.0 Off |                  Off |
| 30%   29C    P8    11W / 230W |    236MiB / 24564MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
```

_Note: Holoviz should support copying data across CUDA devices in a future release._

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
