# Use both Integrated and Discrete GPUs on NVIDIA Developer Kits

NVIDIA Developer Kits like the [NVIDIA IGX Orin](https://www.nvidia.com/en-us/edge-computing/products/igx/) or the [NVIDIA Clara AGX](https://www.nvidia.com/en-gb/clara/intelligent-medical-instruments/) have both a discrete GPU (dGPU - optional on IGX Orin) and an integrated GPU (iGPU - Tegra SoC).

As of this release, when these developer kits are flashed to leverage the dGPU, there are two limiting factors preventing the use of the iGPU:

1. Conflict between the dGPU kernel mode driver and the iGPU display kernel driver (both named `nvidia.ko`). This conflict is not addressable at this time, meaning that **the iGPU cannot be used for display while the dGPU is enabled**.
2. Conflicts between the user mode driver libraries (ex: `libcuda.so`) and the compute stack (ex: `libcuda_rt.so`) for dGPU and iGPU.

We provide utilities to work around the second conflict:

`````{tab-set}
````{tab-item} IGX SW 1.0

1. From an IGX developer kit flashed for dGPU, run the following command to enable iGPU container support:

   ```bash
   sudo /opt/nvidia/l4t-igpu-container-on-dgpu-host-config/l4t-igpu-container-on-dgpu-host-config.sh configure
   ```

  Refer to the [IGX user guide][igx-igpu-dgpu] for details.

2. To leverage both GPUs in Holoscan, you can either:

   1. create separate Holoscan applications running concurrently, where the iGPU application must run in the Holoscan iGPU container, and the dGPU application can run bare metal or in the Holoscan dGPU container. Refer to the [IGX user guide][igx-igpu-dgpu] for details on how to launch a Holoscan container using the iGPU.
   2. create a single distributed application that leverages both the iGPU and dGPU by executing separate fragments on the iGPU and on the dGPU.

[igx-igpu-dgpu]: https://docs.nvidia.com/igx-orin/user-guide/latest/igpu-dgpu.html

The example below shows the ping distributed application between the iGPU and dGPU using Holoscan containers:

```bash
COMMON_DOCKER_FLAGS="--rm -i --init --net=host
--runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=all
--cap-add CAP_SYS_PTRACE --ipc=host --ulimit memlock=-1 --ulimit stack=67108864
"
HOLOSCAN_VERSION=3.3.0
HOLOSCAN_IMG="nvcr.io/nvidia/clara-holoscan/holoscan:v$HOLOSCAN_VERSION"
HOLOSCAN_DGPU_IMG="$HOLOSCAN_IMG-dgpu"
HOLOSCAN_IGPU_IMG="$HOLOSCAN_IMG-igpu"

# Pull images
docker pull $HOLOSCAN_DGPU_IMG
docker pull $HOLOSCAN_IGPU_IMG

# Run ping distributed (python) in dGPU container
# - Making this one the `driver`, but could be igpu too
# - Using & to not block the terminal to run igpu afterwards. Could run igpu in separate terminal instead.
docker run \
  $COMMON_DOCKER_FLAGS \
  $HOLOSCAN_DGPU_IMG \
  bash -c "python3 ./examples/ping_distributed/python/ping_distributed.py --gpu --worker --driver" &

# Run ping distributed (c++) in iGPU container
docker run \
  $COMMON_DOCKER_FLAGS \
  -e NVIDIA_VISIBLE_DEVICES=nvidia.com/igpu=0 \
  $HOLOSCAN_IGPU_IMG \
  bash -c "./examples/ping_distributed/cpp/ping_distributed --gpu --worker"
```

````
````{tab-item} HoloPack 1.2+
The [L4T Compute Assist](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/l4t-compute-assist) is a container on NGC which isolates the iGPU stack by containing the L4T BSP packages in order to enable iGPU compute on the developer kits configured for dGPU. Other applications can run concurrently on the dGPU, natively or in another container.
````
`````

:::{attention}
These utilities enable using the iGPU for capabilities other than **display** only, since they do not address the first conflict listed above.
:::
