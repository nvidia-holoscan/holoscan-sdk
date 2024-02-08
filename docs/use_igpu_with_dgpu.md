# Use both Integrated and Discrete GPUs on Holoscan developer kits

Holoscan developer kits like the [NVIDIA IGX Orin](https://www.nvidia.com/en-us/edge-computing/products/igx/) or the [NVIDIA Clara AGX](https://www.nvidia.com/en-gb/clara/intelligent-medical-instruments/) have both a discrete GPU (dGPU - optional on IGX Orin) and an integrated GPU (iGPU - Tegra SoC). At this time, when these developer kits are flashed (using HoloPack) to leverage the discrete GPU, the integrated GPU cannot be used due to conflicts between the CUDA libraries for dGPU and iGPU stack.

Starting with the Holoscan SDK 0.5, we provide a utility container on NGC named [L4T Compute Assist](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/l4t-compute-assist) which isolates the iGPU stack in order to enable iGPU compute on the developer kits configured for dGPU. Other applications can run concurrently on the dGPU, natively or in another container.

:::{attention}
This container enables using the iGPU for compute capabilities only (not graphics).
:::
