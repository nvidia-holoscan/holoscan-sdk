# Holoscan SDK

The **Holoscan SDK** is part of [NVIDIA Holoscan](https://github.com/nvidia-holoscan), the AI sensor processing platform that combines hardware systems for low-latency sensor and network connectivity, optimized libraries for data processing and AI, and core microservices to run streaming, imaging, and other applications, from embedded to edge to cloud. It can be used to build streaming AI pipelines for a variety of domains, including Medical Devices, High Performance Computing at the Edge, Industrial Inspection and more.

> In previous releases, the prefix [`Clara`](https://developer.nvidia.com/industries/healthcare) was used to define Holoscan as a platform designed initially for [medical devices](https://www.nvidia.com/en-us/clara/developer-kits/). As Holoscan has grown, its potential to serve other areas has become apparent. With version 0.4.0, we're proud to announce that the Holoscan SDK is now officially built to be domain-agnostic and can be used to build sensor AI applications in multiple domains. Note that some of the content of the SDK (sample applications) or the documentation might still appear to be healthcare-specific pending additional updates. Going forward, domain specific content will be hosted on the [HoloHub](https://github.com/nvidia-holoscan/holohub) repository.

Visit the [NGC demo website](https://demos.ngc.nvidia.com/holoscan) for a live demonstration of some of Holoscan capabilities.

## Table of Contents
- [Documentation](#documentation)
- [Prerequisites](#prerequisites)
  - [For Clara AGX and NVIDIA IGX Orin Developer Kits (aarch64)](#for-clara-agx-and-nvidia-igx-orin-developer-kits-aarch64)
  - [For x86_64 systems](#for-x86_64-systems)
- [Using the released SDK](#using-the-released-sdk)
- [Building the SDK from source](#building-the-sdk-from-source)
  - [Recommended: using the `run` script](#recommended-using-the-run-script)
  - [Cross-compilation](#cross-compilation)
  - [Advanced: Docker + CMake](#advanced-docker-cmake)
  - [Advanced: Local environment + CMake](#advanced-local-environment-cmake)
- [Utilities](#utilities)
  - [Code coverage](#code-coverage)
- [Troubleshooting](#troubleshooting)
- [Repository structure](#repository-structure)

## Documentation

* The latest SDK user guide is available at https://docs.nvidia.com/clara-holoscan.
* For a full list of Holoscan documentation, visit the [Holoscan developer page](https://developer.nvidia.com/clara-holoscan-sdk).

## Prerequisites

The Holoscan SDK currently supports the [Holoscan Developer Kits](https://www.nvidia.com/en-us/clara/developer-kits) (aarch64) as well as x86_64 systems.

### For Clara AGX and NVIDIA IGX Orin Developer Kits (aarch64)

Set up your developer kit:
- [Clara AGX Developer Kit User Guide](https://developer.nvidia.com/clara-agx-developer-kit-user-guide), or
- [NVIDIA IGX Orin Developer Kit User Guide](https://developer.nvidia.com/igx-orin-developer-kit-user-guide).

> Make sure you have joined the [Holoscan SDK Program](https://developer.nvidia.com/clara-holoscan-sdk-program) and, if needed, the [Rivermax SDK Program](https://developer.nvidia.com/nvidia-rivermax-sdk) before using the NVIDIA SDK Manager.

[SDK Manager](https://docs.nvidia.com/sdk-manager/install-with-sdkm-clara/) will install **Holopack 1.1** as well as the `nvgpuswitch.py` script. Once configured for dGPU mode, your developer kit will include the following necessary components to build the SDK:
- [NVIDIA Jetson Linux](https://developer.nvidia.com/embedded/jetson-linux): 34.1.2
- [NVIDIA dGPU drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes): 510.73.08
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (for containerized development)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit): 11.6.1 (for local development only)
- [TensorRT](https://developer.nvidia.com/tensorrt): 8.2.3 (for local development only)
- Optional Rivermax support
  - [OFED Network Drivers](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/): 5.7
  - [Rivermax SDK](https://developer.nvidia.com/networking/rivermax): 1.11 (for local development only)
  - *Note: [GPUDirect](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/additional_setup.html#setting-up-gpudirect-rdma) drivers (required for Emergent support) need to be installed manually at this time*

Refer to the user guide for additional steps needed to support specific technologies, such as [AJA cards](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/aja_setup.html) or [Emergent cameras](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/emergent_setup.html).

> Additional dependencies are required when developing locally instead of using a containerized environment, see details in the [section below](#advanced-local-environment--cmake).

### For x86_64 systems

You'll need the following to build applications from source on x86_64:
- OS: Ubuntu 20.04
- NVIDIA GPU
  - Ampere or above recommended for best performance
  - [Quadro/NVIDIA RTX](https://www.nvidia.com/en-gb/design-visualization/desktop-graphics/) necessary for [GPUDirect RDMA](https://developer.nvidia.com/gpudirect) support
- [NVIDIA dGPU drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes): 510.73.08
- For containerized development (recommended):
  - [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- For local development (advanced):
  - See details in the [section below](#advanced-local-environment--cmake).
- For Rivermax support (optional):
  - [NVIDIA ConnectX SmartNIC](https://www.nvidia.com/en-us/networking/ethernet-adapters/)
  - [OFED Network Drivers](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/): 5.7
  - [Rivermax SDK](https://developer.nvidia.com/networking/rivermax): 1.11 (for local development only)
  - [GPUDirect Drivers](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/additional_setup.html#setting-up-gpudirect-rdma)

## Using the released SDK

The Holoscan SDK is available as part of the following packages:
- 🐋 The [Holoscan container image on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan) includes the Holoscan libraries, GXF extensions, headers, example source code, and sample datasets, as well as all the dependencies that were tested with Holoscan. It is the recommended way to run sample streaming applications, while still allowing you to create your own C++ and Python Holoscan application.
- 🐍 The [Holoscan python wheels on PyPI](https://pypi.org/project/holoscan/) (**NEW IN 0.4**) are the ideal way for Python developers to get started with the SDK, simply using `pip install holoscan`. The wheels include the necessary libraries and extensions, not including example code, built applications, nor sample datasets.
- 📦️ The [Holoscan Debian package on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_dev_deb) (**NEW IN 0.4**) includes the libraries, headers, and CMake configurations needed for both C++ and Python developers. It does not include example code, pre-built applications, nor sample datasets.

## Building the SDK from source

Follow the instructions below if you want to build the Holoscan SDK yourself.

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

**Run the applications or examples** inside the container by running their respective commands listed within each directory README file:
  * [reference applications](./apps)
  * [examples](./examples)

### Cross-compilation

While the development Dockerfile does not currently support true cross-compilation, you can compile the Holoscan SDK for the developer kits (arm64) from a x86_64 host using an emulation environment.

1. [Install qemu](https://docs.nvidia.com/datacenter/cloud-native/playground/x-arch.html#emulation-environment)
2. Export your target arch: `export HOLOSCAN_BUILD_PLATFORM=linux/arm64`
3. Clear your build cache: `./run clear_cache`
4. Rebuild: `./run build`

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

The commands to run the **[reference applications](./apps)** or the **[examples](./examples)** are then the same as in the dockerized environment, and can be found in the respective source directory READMEs.

## Utilities

### Code coverage
To generate a code coverage report use the following command after completing setup:
```sh
./run coverage
```

Open the file build/coverage/index.html to access the interactive coverage web tool.

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


## Repository structure

The repository is organized as such:
- `apps/`: source code for the sample applications
- `cmake/`: CMake configuration files
- `examples/`: source code for the examples
- `gxf_extensions/`: source code for the holoscan SDK gxf codelets
- `include/`: source code for the holoscan SDK
- `modules/`: source code for the holoscan SDK modules
- `scripts/`: utility scripts
- `src/`: source code for the holoscan SDK
- `tests/`: tests for the holoscan SDK
