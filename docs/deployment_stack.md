# Deployment Software Stack

NVIDIA Holoscan accelerates deployment of production-quality applications
by providing a set of **OpenEmbedded** build recipes and reference configurations
that can be leveraged to customize and build Holoscan-compatible Linux4Tegra (L4T)
embedded board support packages (BSP) on the NVIDIA IGX Developer Kits.

[Holoscan OpenEmbedded/Yocto recipes](https://github.com/nvidia-holoscan/meta-tegra-holoscan) add
OpenEmbedded recipes and sample build configurations to build BSPs for the NVIDIA IGX Developer Kit
that feature support for discrete GPUs (dGPU), AJA Video Systems I/O boards, and the Holoscan
SDK.
These BSPs are built on a developer's host machine and are then flashed onto the NVIDIA IGX
Developer Kit using provided scripts.

There are two options available to set up a build environment and start building Holoscan BSP images using OpenEmbedded.

1. Use an OpenEmbedded/Yocto build container by building the image from the [meta-tegra-holoscan `env` directory](https://github.com/nvidia-holoscan/meta-tegra-holoscan/tree/main/env#readme). The `holoscan-oe-builder` image previously published on NGC has been retired; follow those instructions to build the container locally and run setup, BitBake, and flash workflows from your workspace.
2. Set up a local build environment in which all dependencies are fetched and installed manually by the developer directly on their host machine.
Please refer to the [Holoscan OpenEmbedded/Yocto recipes README](https://github.com/nvidia-holoscan/meta-tegra-holoscan/blob/main/README.md) for more information on how to use the local build environment.
