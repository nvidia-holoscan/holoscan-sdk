# Deployment Software Stack

NVIDIA Holoscan accelerates deployment of production-quality applications
by providing a set of **OpenEmbedded** build recipes and reference configurations
that can be leveraged to customize and build Holoscan-compatible Linux4Tegra (L4T)
embedded board support packages (BSP) on Holoscan Developer Kits.

[Holoscan OpenEmbedded/Yocto recipes](https://github.com/nvidia-holoscan/meta-tegra-holoscan) add
OpenEmbedded recipes and sample build configurations to build BSPs for NVIDIA Holoscan Developer Kits
that feature support for discrete GPUs (dGPU), AJA Video Systems I/O boards, and the Holoscan
SDK.
These BSPs are built on a developer's host machine and are then flashed onto a Holoscan Developer Kit
using provided scripts.

There are two options available to set up a build environment and start
building Holoscan BSP images using OpenEmbedded.

- The first sets up a local build environment in which all dependencies
are fetched and installed manually by the developer directly on their host machine.
Please refer to the [Holoscan OpenEmbedded/Yocto recipes README](https://github.com/nvidia-holoscan/meta-tegra-holoscan/blob/main/README.md)
for more information on how to use the local build environment.
- The second uses a [Holoscan OpenEmbedded/Yocto Build Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan-oe-builder) that is provided by NVIDIA on NGC which contains all of
the dependencies and  configuration scripts such that the entire process of
building and flashing a BSP can be done with just a few simple commands.
