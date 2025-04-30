(clara_holoscan_setup)=

# SDK Installation

The section below refers to the installation of the Holoscan SDK referred to as the **development stack**, designed for NVIDIA Developer Kits (arm64), and for x86_64 Linux compute platforms, ideal for development and testing of the SDK.

:::{note}
or Holoscan Developer Kits such as the [IGX Orin Developer Kit](https://www.nvidia.com/en-us/edge-computing/products/igx/), an alternative option is the [deployment stack](./deployment_stack.md), based on [OpenEmbedded](https://www.openembedded.org/wiki/Main_Page) ([Yocto](https://www.yoctoproject.org/) build system) instead of Ubuntu. This is recommended to limit your stack to the software components strictly required to run your Holoscan application. The runtime Board Support Package (BSP) can be optimized with respect to memory usage, speed, security and power requirements.
:::

## Prerequisites

`````{tab-set}
````{tab-item} NVIDIA Developer Kits

Setup your developer kit:

Developer Kit | User Guide | OS | GPU Mode
------------- | ---------- | --- | ---
[NVIDIA IGX Orin][igx] | [Guide][igx-guide] | [IGX Software][igx-sw] 1.1 Production Release | iGPU **or*** dGPU
[NVIDIA Jetson AGX Orin and Orin Nano][jetson-orin] | [Guide][jetson-guide] | [JetPack][jp] 6.1 | iGPU
[NVIDIA Clara AGX][clara-agx]<br>_Only supporting the NGC container_ | [Guide][clara-guide] | [HoloPack][sdkm] 1.2<br>_[Upgrade to 535+ drivers required][cagx-upgrade]_ | dGPU

[clara-agx]: https://www.nvidia.com/en-gb/clara/intelligent-medical-instruments
[clara-guide]: https://github.com/nvidia-holoscan/holoscan-docs/blob/main/devkits/clara-agx/clara_agx_user_guide.md
[cagx-upgrade]: https://github.com/nvidia-holoscan/holoscan-docs/blob/main/devkits/clara-agx/clara_agx_user_guide.md#update-nvidia-drivers
[sdkm]: https://developer.nvidia.com/drive/sdk-manager
[igx]: https://www.nvidia.com/en-us/edge-computing/products/igx/
[igx-guide]: https://developer.nvidia.com/igx-orin-developer-kit-user-guide
[igx-sw]: https://developer.nvidia.com/igx-downloads
[meta-tegra]: https://github.com/nvidia-holoscan/meta-tegra-holoscan
[jetson-orin]: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/
[jetson-guide]: https://developer.nvidia.com/embedded/learn/jetson-agx-orin-devkit-user-guide/index.html
[jp]: https://developer.nvidia.com/embedded/jetpack

<sup>_* iGPU and dGPU can be used concurrently on a single developer kit in dGPU mode. See [details here](./use_igpu_with_dgpu.md)._</sup>

````
````{tab-item} NVIDIA SuperChips

This version of the Holoscan SDK was tested on the Grace-Hopper SuperChip (GH200) with Ubuntu 22.04. Follow setup instructions [here](https://docs.nvidia.com/grace-ubuntu-install-guide.pdf).

:::{attention}
Display is not supported on SBSA/superchips. You can however do headless rendering with [HoloViz](./visualization.md#holoviz-operator) for example.
:::

````
````{tab-item} x86_64 Workstations

Supported x86_64 distributions:

OS | NGC Container | Debian/RPM package | Python wheel | Build from source
-- | ------------- | -------------- | ------------ | -----------------
**Ubuntu 22.04** | Yes | Yes | Yes | Yes
**RHEL 9.x** | Yes | No | No | No¹
**Other Linux distros** | No² | No | No³ | No¹

¹ <sup>Not formally tested or supported, but expected to work if building bare metal with the adequate dependencies.</sup><br>
² <sup>Not formally tested or supported, but expected to work if [supported by the NVIDIA container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/supported-platforms.html).</sup><br>
³ <sup>Not formally tested or supported, but expected to work if the glibc version of the distribution is 2.35 or above.</sup><br>

NVIDIA discrete GPU (dGPU) requirements:
- Ampere or above recommended for best performance
- [Quadro/NVIDIA RTX](https://www.nvidia.com/en-gb/design-visualization/desktop-graphics/) necessary for GPUDirect RDMA support
- Tested with [NVIDIA Quadro RTX 6000](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/quadro-rtx-6000-us-nvidia-704093-r4-web.pdf) and [NVIDIA RTX A6000](https://www.nvidia.com/en-us/design-visualization/rtx-a6000/)
- [NVIDIA dGPU drivers](https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes): 535 or above

````
`````

- For RDMA Support, follow the instructions in the [Enabling RDMA](./set_up_gpudirect_rdma.md) section.
- Additional software dependencies might be needed based on how you choose to install the SDK (see section below).
- Refer to the [Additional Setup](./additional_setup.md) and [Third-Party Hardware Setup](./third_party_hw_setup.md) sections for additional prerequisites.

## Install the SDK

We provide multiple ways to install and run the Holoscan SDK:

### Instructions

`````{tab-set}
````{tab-item} NGC Container
- **dGPU** (x86_64, IGX Orin dGPU, Clara AGX dGPU, GH200)
   ```bash
   docker pull nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-dgpu
   ```
- **iGPU** (Jetson, IGX Orin iGPU, Clara AGX iGPU)
   ```bash
   docker pull nvcr.io/nvidia/clara-holoscan/holoscan:v3.2.0-igpu
   ```
See details and usage instructions on [NGC][container].
````
````{tab-item} Debian package

Try the following to install the holoscan SDK:
```sh
sudo apt update
sudo apt install holoscan
```

:::{attention}
This will not install dependencies needed for the Torch nor ONNXRuntime inference backends. To do so, install transitive dependencies by adding the `--install-suggests` flag to `apt install holoscan`, and refer to the support matrix below for links to install libtorch and onnxruntime.
:::

#### Troubleshooting

**If `holoscan` is not found with apt:**

```txt
E: Unable to locate package holoscan
```

Try the following before repeating the installation steps above:

- **IGX Orin**: Ensure the [compute stack is properly installed](https://docs.nvidia.com/igx-orin/user-guide/latest/base-os.html#installing-the-compute-stack) which should configure the L4T repository source. If you still cannot install the Holoscan SDK, use the [`arm64-sbsa` installer](https://developer.nvidia.com/holoscan-downloads?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) from the CUDA repository.
- **Jetson**: Ensure [JetPack is properly installed](https://developer.nvidia.com/embedded/jetpack) which should configure the L4T repository source.  If you still cannot install the Holoscan SDK, use the [`aarch64-jetson` installer](https://developer.nvidia.com/holoscan-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) from the CUDA repository.
- **GH200**: Use the [`arm64-sbsa` installer](https://developer.nvidia.com/holoscan-downloads?target_os=Linux&target_arch=arm64-sbsa&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) from the CUDA repository.
- **x86_64**: Use the [`x86_64` installer](https://developer.nvidia.com/holoscan-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network) from the CUDA repository.

---

**If you get missing CUDA libraries at runtime like below:**

```log
ImportError: libcudart.so.12: cannot open shared object file: No such file or directory
```

This could happen if your system has multiple CUDA Toolkit component versions installed. Find the path of the missing CUDA library (`libcudart.so.12` here) using `find /usr/local/cuda* -name libcudart.so.12` and select that path in `sudo update-alternatives --config cuda`. If that library is not found, or other cuda toolkit libraries become missing afterwards, you could try a clean reinstall of the full CUDA Toolkit:

```bash
sudo apt update && sudo apt install -y cuda-toolkit-12-6
```

---

**If you get missing CUDA headers at compile time like below:**

```cmake
the link interface contains: CUDA::nppidei but the target was not found. [...] fatal error: npp.h: No such file or directory
```

Generally the same issue as above due from mixing CUDA Toolkit component versions in your environment. Confirm the path of the missing CUDA header (`npp.h` here) with `find /usr/local/cuda-* -name npp.h` and follow the same instructions as above.

---

**If you get missing TensorRT libraries at runtime like below:**

```
Error: libnvinfer.so.8: cannot open shared object file: No such file or directory
...
Error: libnvonnxparser.so.8: cannot open shared object file: No such file or directory
```

This could happen if your system has a different major version installed than version 8. Try to reinstall TensorRT 8 with:

```bash
sudo apt update && sudo apt install -y libnvinfer-bin="8.6.*"
```

---

**If you cannot import the holoscan Python module:**

```sh
ModuleNotFoundError: No module named 'holoscan'
```

Python support is removed from the Holoscan SDK Debian package as of v3.0.0. Please install the Holoscan Python wheel.

````
````{tab-item} Python wheel
```bash
pip install holoscan
```
See details and troubleshooting on [PyPI][pypi].

:::{note}
For x86_64, ensure that the [CUDA Toolkit is installed](https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network).
:::

````
`````

[container]: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/containers/holoscan
[pypi]: https://pypi.org/project/holoscan

### Not sure what to choose?

- The [**Holoscan container image on NGC**][container] it the safest way to ensure all the dependencies are present with the expected versions (including Torch and ONNX Runtime), and should work on most Linux distributions. It is the simplest way to run the embedded examples, while still allowing you to create your own C++ and Python Holoscan application on top of it. These benefits come at a cost:
  - large image size from the numerous (some of them optional) dependencies. If you need a lean runtime image, see {ref}`section below<runtime-container>`.
  - standard inconvenience that exist when using Docker, such as more complex run instructions for proper configuration.
- If you are confident in your ability to manage dependencies on your own in your host environment, the **Holoscan Debian package** should provide all the capabilities needed to use the Holoscan SDK, assuming you are on Ubuntu 22.04.
- If you are not interested in the C++ API but just need to work in Python, or want to use a different version than Python 3.10, you can use the [**Holoscan python wheels**][pypi] on PyPI. While they are the easiest solution to install the SDK, it might require the most work to setup your environment with extra dependencies based on your needs. Finally, they are only formally supported on Ubuntu 22.04, though should support other linux distributions with glibc 2.35 or above.

|  | NGC dev Container | Debian Package | Python Wheels |
|---|:---:|:---:|:---:|
| Runtime libraries | **Included** | **Included** | **Included** |
| Python module | 3.10 | 3.10 | **3.9 to 3.12** |
| C++ headers and<br>CMake config | **Included** | **Included** | N/A |
| Examples (+ source) | **Included** | **Included** | [retrieve from<br>GitHub][examples] |
| Sample datasets | **Included** | [retrieve from<br>NGC][data] | [retrieve from<br>NGC][data] |
| CUDA runtime [^1] | **Included** | automatically [^2]<br>installed | require manual<br>installation |
| [NPP][npp] support [^3] | **Included** | automatically [^2]<br>installed | require manual<br>installation |
| [TensorRT][trt] support [^4] | **Included** | automatically [^2]<br>installed | require manual<br>installation |
| [Vulkan][vulkan] support [^5] | **Included** | automatically [^2]<br>installed | require manual<br>installation |
| [V4L2][v4l2] support [^6] | **Included** | automatically [^2]<br>installed | require manual<br>installation |
| [Torch][torch] support [^7] | **Included** | require manual [^8]<br>installation | require manual [^8]<br>installation |
| [ONNX Runtime][ort] support [^9] | **Included** | require manual [^10]<br>installation | require manual [^10]<br>installation |
| [MOFED][mofed] support [^11] | **User space included** <br>Install kernel drivers on the host | require manual <br>installation | require manual <br>installation |
| [CLI] support | [require manual installation](./holoscan_packager.md#cli-installation) | [require manual installation](./holoscan_packager.md#cli-installation) | [require manual installation](./holoscan_packager.md#cli-installation) |

[examples]: https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples#readme
[data]: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara-holoscan/collections/clara_holoscan
[npp]: https://developer.nvidia.com/npp
[trt]: https://developer.nvidia.com/tensorrt
[vulkan]: https://developer.nvidia.com/vulkan
[v4l2]: https://en.wikipedia.org/wiki/Video4Linux
[torch]: https://pytorch.org/
[ort]: https://onnxruntime.ai/
[mofed]: https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/
[cli]: ./holoscan_packager.md
[^1]: [CUDA 12](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-installation-guide-linux/index.html) is required. Already installed on NVIDIA developer kits with IGX Software and JetPack.
[^2]: Debian installation on x86_64 requires the [latest cuda-keyring package](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu) to automatically install all dependencies.
[^3]: NPP 12 needed for the FormatConverter and BayerDemosaic operators. Already installed on NVIDIA developer kits with IGX Software and JetPack.
[^4]: TensorRT 10.3+ needed for the Inference operator. Already installed on NVIDIA developer kits with IGX Software and JetPack.
[^5]: Vulkan 1.3.204+ loader needed for the HoloViz operator (+ libegl1 for headless rendering). Already installed on NVIDIA developer kits with IGX Software and JetPack.
[^6]: V4L2 1.22+ needed for the V4L2 operator. Already installed on NVIDIA developer kits with IGX Software and JetPack.  V4L2 also requires libjpeg.
[^7]: Torch support requires LibTorch 2.5+, TorchVision 0.20+, OpenBLAS 0.3.20+, OpenMPI v4.1.7a1+, UCC 1.4+, MKL 2021.1.1 (x86_64 only), NVIDIA Performance Libraries (aarch64 dGPU only), libpng, and libjpeg. Note that container builds use OpenMPI and UCC originating from the NVIDIA HPC-X package bundle.
[^8]: To install LibTorch and TorchVision, either build them from source, download our [pre-built packages](https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/), or copy them from the holoscan container (in `/opt`).
[^9]: ONNXRuntime 1.18.1+ needed for the Inference operator. Note that ONNX models are also supported through the TensorRT backend of the Inference Operator.
[^10]: To install ONNXRuntime, either build it from source, download our [pre-built package](https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/) with CUDA 12 and TensoRT execution provider support, or copy it from the holoscan container (in `/opt/onnxruntime`).
[^11]: Tested with MOFED 24.07

### Need more control over the SDK?

The [Holoscan SDK source repository](https://github.com/nvidia-holoscan/holoscan-sdk) is **open-source** and provides reference implementations, as well as infrastructure, for building the SDK yourself.

:::{attention}
We only recommend building the SDK from source if you need to build it with debug symbols or other options not used as part of the published packages. If you want to write your own operator or application, you can use the SDK as a dependency (and contribute to [HoloHub](https://github.com/nvidia-holoscan/holohub)). If you need to make other modifications to the SDK, [file a feature or bug request](https://forums.developer.nvidia.com/c/healthcare/holoscan-sdk/320/all).
:::

(runtime-container)=

### Looking for a light runtime container image?

The current Holoscan container on NGC has a large size due to including all the dependencies for each of the built-in operators, but also because of the development tools and libraries that are included. Follow the [instructions on GitHub](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/DEVELOP.md#runtime-container) to build a runtime container without these development packages. This page also includes detailed documentation to assist you in only including runtime dependencies your Holoscan application might need.
