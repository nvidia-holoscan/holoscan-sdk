# Use both Integrated and Discrete GPUs on NVIDIA Developer Kits

NVIDIA Developer Kits like the [NVIDIA IGX Orin](https://www.nvidia.com/en-us/edge-computing/products/igx/) or the [NVIDIA Clara AGX](https://www.nvidia.com/en-gb/clara/intelligent-medical-instruments/) have both a discrete GPU (dGPU - optional on IGX Orin) and an integrated GPU (iGPU - Tegra SoC).

As of this release, when these developer kits are flashed to leverage the dGPU, there are two limiting factors preventing the use of the iGPU:

1. Conflict between the dGPU kernel mode driver and the iGPU display kernel driver (both named `nvidia.ko`). This conflict is not addressable at this time, meaning that **the iGPU cannot be used for display while the dGPU is enabled**.
2. Conflicts between the user mode driver libraries (ex: `libcuda.so`) and the compute stack (ex: `libcuda_rt.so`) for dGPU and iGPU.

We provide utilities to work around the second conflict:

`````{tab-set}
````{tab-item} IGX SW 1.0 DP
An improved solution will be introduced alongside the IGX SW 1.0 GA release.
````
````{tab-item} HoloPack 1.2+
The [L4T Compute Assist](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/l4t-compute-assist) is a container on NGC which isolates the iGPU stack in order to enable iGPU compute on the developer kits configured for dGPU. Other applications can run concurrently on the dGPU, natively or in another container.
````
`````

:::{attention}
These utilities enable using the iGPU for capabilities other than **display** only, since they do not address the first conflict listed above.
:::
