# Clara Holoscan Embedded SDK

[NVIDIA Clara™ Holoscan](https://github.com/nvidia/clara-holoscan) is the AI computing platform for medical devices, consisting of [Clara Developer Kits](https://www.nvidia.com/en-us/clara/developer-kits/) and the [Clara Holoscan SDK](https://developer.nvidia.com/clara-holoscan-sdk). Clara Holoscan enables medical device developers to create the next-generation of AI-enabled medical devices.

The Clara Holoscan SDK currently supports embedded devices (arm64) as well as x86 (amd64) systems.

## Table of Contents
- [Clara Holoscan Embedded SDK](#clara-holoscan-embedded-sdk)
  - [Table of Contents](#table-of-contents)
  - [User Guide](#user-guide)
  - [Prerequisites](#prerequisites)
  - [Sample applications](#sample-applications)
    - [Endoscopy Tool Tracking](#endoscopy-tool-tracking)
    - [Ultrasound Bone Scoliosis Segmentation](#ultrasound-bone-scoliosis-segmentation)
    - [Hi-Speed Endoscopy](#hi-speed-endoscopy)
  - [Running Sample Applications](#running-sample-applications)
  - [Developing with Holoscan SDK](#developing-with-holoscan-sdk)
    - [Sample data](#sample-data)
    - [Using a development container](#using-a-development-container)
    - [Building and running C++ API-based Endoscopy application](#building-and-running-c-api-based-endoscopy-application)
    - [Local environment](#local-environment)
  - [Troubleshooting](#troubleshooting)
    - [X11: Failed to open display :0 [...] Failed to initialize GLFW](#x11-failed-to-open-display-0--failed-to-initialize-glfw)
    - [GLX: Failed to create context: GLXBadFBConfig](#glx-failed-to-create-context-glxbadfbconfig)
    - [`GXF_ENTITY_COMPONENT_NOT_FOUND` or `GXF_ENTITY_NOT_FOUND`](#gxf_entity_component_not_found-or-gxf_entity_not_found)
    - [No receiver connected to transmitter of <scheduling term id> of entity <x>. The entity will never tick.](#no-receiver-connected-to-transmitter-of-scheduling-term-id-of-entity-x-the-entity-will-never-tick)
    - [AJA device errors](#aja-device-errors)
    - [GXF format converter errors](#gxf-format-converter-errors)
    - [Data download error](#data-download-error)
  - [Repository structure](#repository-structure)

## User Guide

The latest SDK user guide is available at https://docs.nvidia.com/clara-holoscan. Before installing the SDK from this GitHub repo, make sure you have followed the [Clara AGX Developer Kit User Guide](https://developer.nvidia.com/clara-agx-developer-kit-user-guide) or the [Clara Holoscan Developer Kit User Guide](https://developer.download.nvidia.com/CLARA/Clara-Holoscan-Developer-Kit-User-Guide.pdf) to set up your development kit.

## Prerequisites

The Clara Holoscan Embedded SDK and its sample applications are designed to run on any of the [Clara Developer Kits](https://www.nvidia.com/en-us/clara/developer-kits/).

Requirements include:
- [NVIDIA Jetson Linux](https://developer.nvidia.com/embedded/jetson-linux): 34.1.2<sup> [1](#holopack)</sup>
- [NVIDIA dGPU drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes): 510.73.08<sup> [2](#switch-script)</sup>
- [CUDA](https://developer.nvidia.com/cuda-toolkit): 11.6.1<sup> [1](#holopack)</sup>
- [CuDNN](https://developer.nvidia.com/cudnn): 8.3.3.40<sup> [1](#holopack)</sup>
- [TensorRT](https://developer.nvidia.com/tensorrt): 8.2.3<sup> [1](#holopack)</sup>
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)<sup> [1](#holopack)</sup>
- For AJA support: AJA drivers (refer to the [user guide](#user-guide))
- [NGC CLI](https://ngc.nvidia.com/setup/installers/cli)

<sup id="holopack">1. included when installing **[Holopack](https://developer.nvidia.com/embedded/jetpack) 1.1** on your Clara Developer Kit with [SDK Manager][sdkm]</sup>
<br>
<sup id="switch-script">2. included when running the `nvgpuswitch` script on your Clara Developer Kit, installed with the [SDK Manager][sdkm]</sup>

[sdkm]: https://docs.nvidia.com/sdk-manager/install-with-sdkm-clara/

Optionally, for testing or cross-compiling to arm64:
  - GNU/Linux x86_64 with kernel version > 3.10
  - NVIDIA GPU: Architecture >= Pascal, Quadro required to enable RDMA

## Sample applications

### Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.
  - `tracking_aja`: uses an AJA capture card for input stream
  - `tracking_replayer`: uses a pre-recorded video as input

### Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model.
  - `segmentation_aja`: uses an AJA capture card for input stream
  - `segmentation_replayer`: uses a pre-recorded video as input

### Hi-Speed Endoscopy

The example app showcases how high resolution cameras can be used to capture the
scene, processed on GPU and displayed at high frame rate using the GXF framework.
This app requires Emergent Vision Technologies camera and a display with high
refresh rate to keep up with camera's framerate.
  - `hi_speed_endoscopy`: uses Emergent Vision Technologies camera for input stream

## Running Sample Applications

The [Clara Holoscan Sample Applications](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/clara_holoscan_sample_runtime) container is the simplest way to run the sample applications as it includes all necessary binaries and datasets, and allows for some customization of the application graph and its parameters.

Refer to the overview of the container on NGC for prerequisites, setup, and run instructions.

> _Note: the sample applications container from NGC does not include build dependencies to update or generate new extensions, or to build new applications with other extensions. Refer to the section below to do this from source._

## Developing with Holoscan SDK

### Sample data

The sample applications rely on ML models and recorded video data, for endoscopy (~430MB) and ultrasound (~30MB), which can be downloaded from NGC in one of two ways:
  1. Using our python utility script (requires [ngc cli](https://ngc.nvidia.com/setup/installers/cli))
      ```sh
      python3 ./scripts/download_sample_data.py # add --help for additional configurations
      ```
  2. Or, from the NGC resource webpages directly:
      - [Sample App Data for AI-based Endoscopy Tool Tracking](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_endoscopy_sample_data)
      - [Sample App Data for AI-based Bone Scoliosis Segmentation](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/resources/holoscan_ultrasound_sample_data)

  _Note: data should be placed under `test_data/endoscopy` and `test_data/ultrasound` to match the paths currently defined in the sample applications. Those paths could easily be modified in the respective application yaml files, located within the `apps/` folder._

You don't need to download the data if you're building from source because CMake build will download sample data by default (See [./apps/CMakeLists.txt](./apps/CMakeLists.txt)).

### Using a development container

1. `git clone` this repository and `cd` in the top folder. That directory will be mounted in the development container.

2. Build the development image. It requires more resources from NGC, so you'll need to provide your ngc api key available from [ngc.nvidia.com/setup](https://ngc.nvidia.com/setup/api-key):

```sh
# This is needed if you want to access non-public resources in NGC
export NGC_CLI_API_KEY=$YOUR_NGC_API_KEY
```

Then, run the following command to build the image.

```sh
export DOCKER_BUILDKIT=1
docker build -t holoscan-sdk-dev .
```

Or, you can easily execute the commands above using `./run build_image` command.

```sh
./run build_image
```

> #### Compilation with qemu for arm64 architecture
> While the development dockerfile does not currently support true cross-compilation, you can compile for the Clara developer kits (arm64) from a x86_64 host using an emulation environment.
>
> Run the following steps on your host:
> 1. Follow the [installation steps for `qemu`](https://docs.nvidia.com/datacenter/cloud-native/playground/x-arch.html#emulation-environment)
> 2. Add `--platform linux/arm64` to the `docker build` command above
>   - Or, you can export the environment variable `HOLOSCAN_BUILD_PLATFORM` (`export HOLOSCAN_BUILD_PLATFORM=linux/arm64`) and use `./run` commands below to build and launch with arm64 image. Please remove cache folders (`./run clear_cache`) before building apps from source code to avoid compilation errors.
> 3. Run the cmake configuration and build below (step 3)
>
> You can then copy the `build`/`install` folder generated by CMake to a developer kit with a configured environment or container and run applications there.

3. Install GXF package (default location is `.cache/gxf`)

```sh
./run install_gxf
```

4. Configure and build the source in the container using CMake

```sh
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v $(pwd):/workspace/holoscan-sdk \
  -e CMAKE_BUILD_TYPE=Release \
  -e CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
  -w /workspace/holoscan-sdk \
  holoscan-sdk-dev \
  bash -c '
    cmake -S . -B build \
    && cmake --build build \
    && cmake --install build --prefix install --component holoscan-embedded-core \
    && cmake --install build --prefix install --component holoscan-embedded-gxf_extensions \
    && cmake --install build --prefix install --component holoscan-embedded-apps \
    && cmake --install build --prefix install --component holoscan-embedded-gxf_libs \
    && cmake --install build --prefix install --component holoscan-embedded-gxf_bins \
    && cmake --install build --prefix install --component holoscan-embedded-dep_libs
  '
```

Or, you can easily execute the commands above using `./run build` command.

```sh
./run build

# Please execute `./run build -h` for more information.
# Please execute `./run build --dryrun` to see the commands that will be executed.
```

In the bash commands above, the following shows the specific commands used with the comment.

- `cmake -S . -B build`
  - Configure the source in the container.
- `cmake --build build`
  - Build the source in the container (`./build` folder). Need only this command if already configured.
- `cmake --install build --prefix install --component holoscan-embedded-core`
  - Create Holoscan Embedded App SDK's CMake package in `./install` folder.
- `cmake --install build --prefix install --component holoscan-embedded-gxf_extensions`
  - Install Holoscan Embedded SDK's GXF extensions in `./install/gxf_extensions` folder.
- `cmake --install build --prefix install --component holoscan-embedded-apps`
  - Install Holoscan Embedded SDK's apps in `./install/apps` folder.
- `cmake --install build --prefix install --component holoscan-embedded-gxf_libs`
  - Install Holoscan Embedded SDK's GXF libraries (libgxf_xxx.so files) in `./install/lib` folder.
- `cmake --install build --prefix install --component holoscan-embedded-gxf_bins`
  - Install Holoscan Embedded SDK's GXF binaries (gxe) in `./install/bin` folder.
- `cmake --install build --prefix install --component holoscan-embedded-dep_libs`
  - Install Holoscan Embedded SDK's dependencies in `./install/libs` folder.
    - Including `libglad.so`, `libglfw.so`, and so on.

> If you encounter errors during CMake build, please try to remove the `build` or/and `install` directory (and cache folders such as `.cache/cpm` and `.cache/ccache`), and try to build again.
> (You can execute `./run clear_cache` to remove cache/build/install folders).

5. Start the container with the options you need (aja or not...).
    * Note: it is currently necessary to run the apps from the top of the build directory to satisfy relative paths to certain extension resources, like shaders and fonts.

**Without AJA capture card installed**
```sh
docker run -it --rm \
  --runtime=nvidia \
  -u $(id -u):$(id -g) \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/share/vulkan:/usr/share/vulkan \
  -v $(pwd)/test_data:/workspace/test_data \
  -v $(pwd):/workspace/holoscan-sdk \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
  -e DISPLAY=$DISPLAY \
  -w /workspace/holoscan-sdk/build \
  holoscan-sdk-dev
```

You can replace `-w /workspace/holoscan-sdk/build` with `-w /workspace/holoscan-sdk/install` to run the apps from the install directory.
(Shared libraries under `install` folder has relative RPATHs so you can copy `install` folder to another location if you want to run the apps from another location.)

You can also use `./run launch` command to launch the container.

```sh
./run launch
# Please execute `./run launch -h` for more information.
# Please execute `./run launch --dryrun` to see the commands that will be executed.
```

**With AJA capture card installed**

Please update `apps/endoscopy_tool_tracking/app_config.yaml` locally first so that it uses AJA capture card as input.

```yaml
source: "aja"
```

```sh
# `--device /dev/ajantv20:/dev/ajantv20` is added to the docker run command
docker run -it --rm \
  --runtime=nvidia \
  -u $(id -u):$(id -g) \
  --device /dev/ajantv20:/dev/ajantv20 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/share/vulkan:/usr/share/vulkan \
  -v $(pwd)/test_data:/workspace/test_data \
  -v $(pwd):/workspace/holoscan-sdk \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
  -e DISPLAY=$DISPLAY \
  -w /workspace/holoscan-sdk/build \
  holoscan-sdk-dev
```

You can replace `-w /workspace/holoscan-sdk/build` with `-w /workspace/holoscan-sdk/install` to run the apps from the install directory.
(Shared libraries under `install` folder has relative RPATHs so you can copy `install` folder to another location if you want to run the apps from another location.)

You can also use `./run launch` command to launch the container.

```sh
./run launch
# Please execute `./run launch install` to launch container with the working directory set to `/workspace/holoscan-sdk/install`.
# Please execute `./run launch -h` for more information.
# Please execute `./run launch --dryrun` to see the commands that will be executed.
```

6. Run the apps inside the container.
    * Note: You can also run those directly, by appending a command to the command above.

```sh
# Source: video sample data
./apps/endoscopy_tool_tracking_gxf/tracking_replayer
./apps/ultrasound_segmentation_gxf/segmentation_replayer

# Source: AJA capture card (requires `--device /dev/ajantv20:/dev/ajantv20` in docker run command)
# Note that the source should be plugged to the Channel 1 of the AJA card
./apps/endoscopy_tool_tracking_gxf/tracking_aja
./apps/ultrasound_segmentation_gxf/segmentation_aja

# C++ API-based Endoscopy application (that wraps the GXF C-API)
# (Source depends on the `source` field of the configuration in
#  `apps/endoscopy_tool_tracking/app_config.yaml`)
LD_LIBRARY_PATH=$(pwd):$(pwd)/lib:$LD_LIBRARY_PATH ./apps/endoscopy_tool_tracking/endoscopy_tool_tracking

# Source: mock AJA capture card input for testing purposes
./apps/endoscopy_tool_tracking_gxf/tracking_mock
./apps/ultrasound_segmentation_gxf/segmentation_mock
```

Without going inside the container, you can also run the apps directly with `./run launch` command:

```sh
# Source: video sample data (you can replace 'build' with 'install' to run the apps from the install directory)
./run launch build apps/endoscopy_tool_tracking_gxf/tracking_replayer
./run launch build apps/ultrasound_segmentation_gxf/segmentation_replayer

# Source: AJA capture card (you can replace 'build' with 'install' to run the apps from the install directory)
./run launch build apps/endoscopy_tool_tracking_gxf/tracking_aja
./run launch build apps/ultrasound_segmentation_gxf/segmentation_aja

# C++ API-based Endoscopy application (that wraps the GXF C-API)
# (Source depends on the `source` field of the configuration in
#  `apps/endoscopy_tool_tracking/app_config.yaml`)
./run launch build bash -c 'LD_LIBRARY_PATH=$(pwd):$(pwd)/lib:$LD_LIBRARY_PATH apps/endoscopy_tool_tracking/endoscopy_tool_tracking'

# Source: mock AJA capture card input for testing purposes
# (you can replace 'build' with 'install' to run the apps from the install directory)
./run launch build apps/endoscopy_tool_tracking_gxf/tracking_mock
./run launch build apps/ultrasound_segmentation_gxf/segmentation_mock
```

### Building and running C++ API-based Endoscopy application

After you have followed the steps above (Step 1 ~ Step 4), you can build and run the C++ API-based Endoscopy application using the CMake package in the `./install` folder.


1. Configure and build the stand-alone C++ API-based Endoscopy application in the container using CMake

Under `examples/holoscan-endoscopy-app` (`HOLOSCAN_APP_PATH`), there is a CMake project that builds the C++ API-based Endoscopy application.

Holoscan SDK package is available in the `./install` folder (`HOLOSCAN_SDK_PATH` is set in CMake configuration).

GXF package is available in the `.cache/gxf` folder. (`GXF_SDK_PATH` is set in CMake configuration).

```sh
docker run -it --rm \
  -u $(id -u):$(id -g) \
  -v $(pwd):/workspace/holoscan-sdk \
  -e CMAKE_BUILD_TYPE=Release \
  -e CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) \
  -e HOLOSCAN_SDK_PATH=/workspace/holoscan-sdk/install \
  -e GXF_SDK_PATH=/workspace/holoscan-sdk/.cache/gxf \
  -e HOLOSCAN_APP_PATH=/workspace/holoscan-sdk/examples/holoscan-endoscopy-app \
  -w /workspace/holoscan-sdk \
  holoscan-sdk-dev \
  bash -c '
    cd ${HOLOSCAN_APP_PATH} \
    && cmake -S . -B build -DHOLOSCAN_SDK_PATH=${HOLOSCAN_SDK_PATH} -DGXF_SDK_PATH=${GXF_SDK_PATH} \
    && cmake --build build
  '
```

You can also use `./run build_example` command to build the C++ API-based Endoscopy application in the container.

```sh
./run build_example
# Please execute `./run build_example install` to launch container with the working directory set to `/workspace/holoscan-sdk/install`.
# Please execute `./run build_example -h` for more information.
# Please execute `./run build_example --dryrun` to see the commands that will be executed.
```

If you encounter errors in cmake, try removing the build/install directory (and cache folders such as `.cache/cpm` and `.cache/ccache`) and trying again.

2. Start the container with the options you need (aja or not...).
    * Note: it is currently necessary to run the apps from the top of the build directory to satisfy relative paths to certain extension resources, like shaders and fonts.

**Without AJA capture card installed (this example always uses video sample data as a source)**

```sh
docker run -it --rm \
  --runtime=nvidia \
  -u $(id -u):$(id -g) \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /usr/share/vulkan:/usr/share/vulkan \
  -v $(pwd)/test_data:/workspace/test_data \
  -v $(pwd):/workspace/holoscan-sdk \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility,display \
  -e DISPLAY=$DISPLAY \
  -e HOLOSCAN_SDK_PATH=/workspace/holoscan-sdk/install \
  -e GXF_SDK_PATH=/workspace/holoscan-sdk/.cache/gxf \
  -e HOLOSCAN_APP_PATH=/workspace/holoscan-sdk/examples/holoscan-endoscopy-app \
  -w /workspace/holoscan-sdk/build \
  holoscan-sdk-dev
```

You can also use `./run launch_example` command to launch the container.

```sh
./run launch_example
# Please execute `./run launch_example -h` for more information.
# Please execute `./run launch_example --dryrun` to see the commands that will be executed.
```

3. Run the apps inside the container.
    * Note: You can also run those directly appended to the command above.

```sh
export LD_LIBRARY_PATH=$(pwd):${HOLOSCAN_SDK_PATH}/lib:$LD_LIBRARY_PATH

${HOLOSCAN_APP_PATH}/build/holoscan-endoscopy-app
```

Without going inside the container, you can also run the app directly with `./run launch` command:

```sh
# You can replace 'build' with 'install' to run the apps from the install directory
./run launch_example build bash -c 'LD_LIBRARY_PATH=$(pwd):${HOLOSCAN_SDK_PATH}/lib:$LD_LIBRARY_PATH ${HOLOSCAN_APP_PATH}/build/holoscan-endoscopy-app'
```

### Local environment

We recommend building and running the sample applications from the development docker container.

If you want to build and run applications from the bare metal machine without docker (e.g., for building and running the high-speed endoscopy application which requires installing kernel modules), please follow below steps for building the source on local environment.

1. Install following dependencies.
```sh
# For GLFW
sudo apt-get install -y libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxkbcommon-dev \
    libxi-dev

# GL/gl.h for cuda_gl_interop.h
sudo apt install -y libgl-dev
```
2. Download [TensorRT 8.4.1 GA](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and follow the installation steps using [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html). If there is exisiting TensorRT, use below commands to purge it and install new version.
```sh
sudo apt-get purge "libnvinfer*"
sudo apt-get purge "nv-tensorrt-repo*"

sudo dpkg -i nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604_1-1_arm64.deb
sudo apt-key add /var/nv-tensorrt-repo-ubuntu2004-cuda11.6-trt8.4.1.5-ga-20220604/9a60d8bf.pub

sudo apt-get update
sudo apt-get install libnvinfer8=8.4.1-1+cuda11.6
sudo apt-get install libnvinfer-plugin8=8.4.1-1+cuda11.6
sudo apt-get install libnvparsers8=8.4.1-1+cuda11.6
sudo apt-get install libnvonnxparsers8=8.4.1-1+cuda11.6
sudo apt-get install python3-libnvinfer=8.4.1-1+cuda11.6
sudo apt-get install libnvinfer-dev=8.4.1-1+cuda11.6
sudo apt-get install libnvinfer-plugin-dev=8.4.1-1+cuda11.6
sudo apt-get install libnvparsers-dev=8.4.1-1+cuda11.6
sudo apt-get install libnvonnxparsers-dev=8.4.1-1+cuda11.6
sudo apt-get install python3-libnvinfer-dev=8.4.1-1+cuda11.6
sudo apt-get install libnvinfer-bin=8.4.1-1+cuda11.6
sudo apt-get install libnvinfer-samples=8.4.1-1+cuda11.6
sudo apt-get install tensorrt=8.4.1.5-1+cuda11.6
```

3. Install CMake 3.22.2
```sh
sudo rm -r \
    /usr/bin/cmake \
    /usr/bin/cpack \
    /usr/bin/ctest
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
    | sudo gpg --dearmor - \
    | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ focal main"
sudo apt update
sudo apt install --no-install-recommends -y \
    cmake-data=3.22.2-0kitware1ubuntu20.04.1 \
    cmake=3.22.2-0kitware1ubuntu20.04.1
sudo rm -rf /var/lib/apt/lists/*
```

4. Install GXF package
```sh
./run install_gxf
```

5. Install Vulkan SDK and dependencies required
```sh
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    libglm-dev cmake libxcb-dri3-0 libxcb-present0 \
    libpciaccess0 libpng-dev libxcb-keysyms1-dev \
    libxcb-dri3-dev libx11-dev g++ gcc libmirclient-dev \
    libwayland-dev libxrandr-dev libxcb-randr0-dev \
    libxcb-ewmh-dev git python python3 bison libx11-xcb-dev \
    liblz4-dev libzstd-dev python3-distutils qt5-default\
    ocaml-core ninja-build pkg-config libxml2-dev wayland-protocols
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libxrandr-dev libxxf86vm-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev libgl-dev

export VULKAN_SDK_VERSION=1.3.216.0
mkdir /tmp/vulkansdk
cd /tmp/vulkansdk
wget --inet4-only -nv --show-progress \
    --progress=dot:giga https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
tar -xzf vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
rm vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz
cd ${VULKAN_SDK_VERSION}
rm -rf x86_64
./vulkansdk shaderc glslang headers loader;
sudo mkdir /opt/vulkansdk
sudo cp -R /tmp/vulkansdk/${VULKAN_SDK_VERSION}/x86_64/ /opt/vulkansdk/${VULKAN_SDK_VERSION}
```

6. Set paths for compiling the Holoviz Visualizer extension and sample apps. It
is recommended to add above paths in `.bashrc` to avoid repetition.
```sh
vi ~/.bashrc
# Add following two lines:
# export VULKAN_SDK_VERSION=1.3.216.0
# export VULKAN_SDK=/opt/vulkansdk/${VULKAN_SDK_VERSION}
# export PATH="$VULKAN_SDK/bin:${PATH}"
source ~/.bashrc
```

7. If using Emergent Vision Technologies camera, install the latest Emergent SDK referred to in the 
[Clara Holoscan User Guide](https://docs.nvidia.com/clara-holoscan/sdk-user-guide/emergent_setup.html#installing-evt-software)
before configuring and building the source.

8. Configure and build the source locally using CMake.
```sh
cmake -S . -B build \
    -D CMAKE_BUILD_TYPE=Release \
    -D CUDAToolkit_ROOT:PATH=/usr/local/cuda \
    -D CMAKE_CUDA_COMPILER:PATH=/usr/local/cuda/bin/nvcc \
    -D HOLOSCAN_BUILD_HI_SPEED_ENDO_APP=ON
cmake --build build -j
```

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



### Data download error
The following error while running the python3 scripts/download_sample_data.py command is not due to a missing file/directory. It may indicate that you have not set up your NGC CLI.

```sh
FileNotFoundError: [Errno 2] No such file or directory: '/media/m2/workspace/clara-holoscan-embedded-sdk-public/test_data/holoscan_ultrasound_sample_data_v20220606' -> '/media/m2/workspace/clara-holoscan-embedded-sdk-public/test_data/ultrasound'
```

## Repository structure

The repository is organized as such:
- `apps/`: yaml files to define the sample applications
- `cmake/`: CMake custom utilities
- `examples/`: example applications that use the SDK in a standalone way
- `gxf_extensions/`: source code the gxf extensions for holoscan codelets
- `include/`: source code for the holoscan embedded app SDK
- `modules/`: source code for the holoscan SDK modules (e.g., `Holoviz`)
- `scripts/`: utility scripts
- `src/`: source code for the holoscan embedded app SDK
- `tests/`: tests for the holoscan embedded app SDK
