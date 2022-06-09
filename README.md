# Clara Holoscan Embedded SDK

NVIDIA Clara™ Holoscan is the AI computing platform for medical devices, consisting of [Clara Developer Kits](https://www.nvidia.com/en-us/clara/developer-kits/) and the [Clara Holoscan SDK](https://developer.nvidia.com/clara-holoscan-sdk). Clara Holoscan enables medical device developers to create the next-generation of AI-enabled medical devices and take it to the market using [Clara Holoscan MGX](https://www.nvidia.com/en-us/clara/medical-grade-devices/).

## Table of Contents
* [User Guide](#user-guide)
* [Reference applications](#reference-applications)
  + [Endoscopy Tool Tracking](#endoscopy-tool-tracking)
  + [Ultrasound Bone Scoliosis Segmentation](#ultrasound-bone-scoliosis-segmentation)
* [Prerequisites](#prerequisites)
* [Running from container](#running-from-container)
* [Running from source](#running-from-source)
  + [Sample data](#sample-data)
  + [Local environment](#local-environment)
  + [Using a development container](#using-a-development-container)
* [Troubleshooting](#troubleshooting)
* [Repository structure](#repository-structure)

## User Guide

The latest SDK user guide is available at https://docs.nvidia.com/clara-holoscan. Before installing the SDK from this GitHub repo, make sure you've followed the Developer Kit [User Guide](https://developer.nvidia.com/clara-agx-developer-kit-user-guide) to set up your devkit.

## Prerequisites

The Clara Holoscan Embedded SDK and its reference applications are designed to run on any of the [Clara Developer Kits](https://www.nvidia.com/en-us/clara/developer-kits/).

Requirements include:
- [NVIDIA Jetson Linux](https://developer.nvidia.com/embedded/jetson-linux): 34.1.2<sup> [1](#jetpack)</sup>
- [NVIDIA dGPU drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes): 510.73.08<sup> [2](#switch-script)</sup>
- [CUDA](https://developer.nvidia.com/cuda-toolkit): 11.6.1<sup> [1](#jetpack)</sup>
- [CuDNN](https://developer.nvidia.com/cudnn): 8.3.3.40<sup> [1](#jetpack)</sup>
- [TensorRT](https://developer.nvidia.com/tensorrt): 8.2.3<sup> [1](#jetpack)</sup>
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)<sup> [1](#jetpack)</sup>
- For AJA support: AJA drivers (refer to the [user guide](#user-guide))
- [NGC CLI](https://ngc.nvidia.com/setup/installers/cli)

<sup id="jetpack">1. included when installing **[JetPack](https://developer.nvidia.com/embedded/jetpack) 5.0 HP1** on your Clara Developer Kit with [SDK Manager][sdkm]</sup>
<br>
<sup id="switch-script">2. included when running the `nvgpuswitch` script on your Clara Developer Kit, installed with the [SDK Manager][sdkm]</sup>

[sdkm]: https://docs.nvidia.com/sdk-manager/install-with-sdkm-clara/

Optionally, for testing or cross-compiling to arm64:
  - GNU/Linux x86_64 with kernel version > 3.10
  - NVIDIA GPU: Architecture >= Pascal, Quadro required to enable RDMA

## Reference applications

### Endoscopy Tool Tracking

Based on a LSTM (long-short term memory) stateful model, these applications demonstrate the use of custom components for tool tracking, including composition and rendering of text, tool position, and mask (as heatmap) combined with the original video stream.
  - `tracking_aja`: uses an AJA capture card for input stream
  - `tracking_replayer`: uses a pre-recorded video as input

### Ultrasound Bone Scoliosis Segmentation

Full workflow including a generic visualization of segmentation results from a spinal scoliosis segmentation model of ultrasound videos. The model used is stateless, so this workflow could be configured to adapt to any vanilla DNN model.
  - `segmentation_aja`: uses an AJA capture card for input stream
  - `segmentation_replayer`: uses a pre-recorded video as input

## Running from container

The [Clara Holoscan Sample Applications](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/clara_holoscan_sample_runtime) container is the simplest way to run the sample applications as it includes all necessary binaries and datasets, and allows for some customization of the application graph and its parameters.

Refer to the overview of the container on NGC for prerequisites, setup, and run instructions.

> _Note: the sample applications container from NGC does not include build dependencies to update or generate new extensions, or to build new applications with other extensions. Refer to the section below to do this from source._

## Running from source

### Sample data

The reference applications rely on ML models and recorded video data, for endoscopy (~430MB) and ultrasound (~30MB), which can be downloaded from NGC in one of two ways:
  1. Using our python utility script (requires [ngc cli](https://ngc.nvidia.com/setup/installers/cli))
      ```sh
      python3 ./scripts/download_sample_data.py # add --help for additional configurations
      ```
  2. Or, from the NGC resource webpages directly:
      - [Sample App Data for AI-based Endoscopy Tool Tracking](https://ngc.nvidia.com/resources/nvidia:clara-holoscan:holoscan_endoscopy_sample_data)
      - [Sample App Data for AI-based Bone Scoliosis Segmentation](https://ngc.nvidia.com/resources/nvidia:clara-holoscan:holoscan_ultrasound_sample_data)

  _Note: data should be placed under `test_data/endoscopy` and `test_data/ultrasound` to match the paths currently defined in the reference applications. Those paths could easily be modified in the respective application yaml files, located within the `apps/` folder._


### Local environment

Building using a local environment is possible but not actively maintained at this time. We recommend looking at how our [development container](Dockerfile) is configured to learn how to setup your own local environment.

### Using a development container

1. `git clone` this repository and `cd` in the top folder. That directory will be mounted in the development container.

2. Build the development image. It requires more resources from NGC, so you'll need to provide your ngc api key available from [ngc.nvidia.com/setup](https://ngc.nvidia.com/setup/api-key):

```sh
export NGC_CLI_API_KEY=$YOUR_NGC_API_KEY
export DOCKER_BUILDKIT=1
docker build \
  --secret id=NGC_CLI_API_KEY \
  -t holoscan-sdk-dev .
```

> #### Cross-compilation: x86_64 to arm64
> While the development dockerfile does not currently support true cross-compilation, you can compile for the Holoscan developer kits (arm64) from a x86_64 host using an emulation environment.
>
> Run the following steps on your host:
> 1. Follow the [installation steps for `qemu`](https://docs.nvidia.com/datacenter/cloud-native/playground/x-arch.html#emulation-environment)
> 2. Add `--platform linux/arm64` to the `docker build` command above
> 3. Run the cmake configuration and build below (step 3)
>
> You can then copy the `build` folder generated by CMake to a developer kit with a configured environment or container and run applications there.

3. Configure and build the source in the container using CMake

```sh
docker run --rm \
  -v $(pwd):/workspace/holoscan-sdk \ # Mount the source dir
  -w /workspace/holoscan-sdk \        # Run in that directory
  -u $(id -u):$(id -g) \              # As current user
  holoscan-sdk-dev \                  # Your newly built image
  bash -c '
    cmake -S . -B build -D CMAKE_BUILD_TYPE=Release \
      -D ajantv2_DIR:PATH=/opt/ajantv2 \
      -D CUDAToolkit_ROOT:PATH=/usr/local/cuda-11.6 \
      -D glad_DIR:PATH=/opt/glad \
      -D glfw3_DIR:PATH=/opt/glfw \
      -D GXF_DIR:PATH=/opt/gxf \
      -D nanovg_DIR:PATH=/opt/nanovg \
      -D TensorRT_DIR:PATH=/opt/tensorrt \
      -D yaml-cpp_DIR:PATH=/opt/yaml-cpp \
    && cmake --build build -j # or just this command if already configured
  '
```
If you encounter errors in cmake, try removing the build directory and trying again.

4. Start the container with the options you need (aja or not...).
    * Note: it is currently necessary to run the apps from the top of the build directory to satisfy relative paths to certain extension resources, like shaders and fonts.

```sh
docker run -it --rm \
  --runtime=nvidia \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,video,compute,utility \  # Access NVIDIA GPU and drivers
  --device /dev/ajantv20:/dev/ajantv20 \                          # Access AJA capture card and drivers
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \          # Access X11 display
  -v $(pwd)/test_data:/workspace/test_data \                      # Access test data
  -v $(pwd)/build:/workspace/holoscan-sdk/build \                 # Access your build artifacts
  -w /workspace/holoscan-sdk/build \                              # Run from the build directory
  -u $(id -u):$(id -g) \                                          # As current user
  holoscan-sdk-dev
```

 5. Run the apps inside the container.
     * Note: You can also run those directly appended to the command above.

```sh
# Source: video sample data
./apps/endoscopy_tool_tracking/tracking_replayer
./apps/ultrasound_segmentation/segmentation_replayer

# Source: AJA capture card
./apps/endoscopy_tool_tracking/tracking_aja
./apps/ultrasound_segmentation/segmentation_aja

# Source: mock AJA capture card input for testing purposes
./apps/endoscopy_tool_tracking/tracking_mock
./apps/ultrasound_segmentation/segmentation_mock
```

## Troubleshooting

### X11: Failed to open display :0 [...] Failed to initialize GLFW

Enable permissions to your X server from Docker, either:
- Passing `-u $(id -u):$(id -g)` to `docker run`, or
- Running `xhost +local:docker` on your host

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
- `apps/`: yaml files to define the reference applications
- `cmake/`: CMake custom utilities
- `gxf_extensions/`: source code the gxf extensions for holoscan codelets
- `scripts/`: utility scripts
- `test/`: testing utilities
