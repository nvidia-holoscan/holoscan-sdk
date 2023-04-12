This folder contains patches that are either applied during the build process of Holoscan SDK
or have been used to build artifacts used by the SDK

# gxf_remove_complex_primitives_support.patch
This patch is applied at build time of the Holoscan SDK and removes complex primitive support from
GXF since this feature does not compile with CUDA 11.6 and C++17.

# v4l2-plugin.patch
This patch is released under GNU LGPL License v2.1

This patch applies to the NVIDIA tegra v4l2 repository
https://nv-tegra.nvidia.com/tegra/v4l2-src/v4l2_libs.git (branch l4t/l4t-r35.2.1)

This patch brings modification to v4l2-plugin to load plugins in the local directory
(where the library is loaded) instead of hard-coded path.
