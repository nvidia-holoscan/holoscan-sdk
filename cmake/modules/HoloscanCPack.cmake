# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copy NOTICE file for packaging
install(FILES "${CMAKE_CURRENT_LIST_DIR}/cpack/NOTICE.txt"
    DESTINATION .
    RENAME NOTICE
    COMPONENT holoscan-cpack
)

# Copy LICENSE file for installation
install(FILES "${CMAKE_BINARY_DIR}/LICENSE.txt"
    DESTINATION "/usr/share/doc/holoscan/"
    RENAME copyright
    COMPONENT holoscan-cpack
)

# CPACK config
set(CPACK_PACKAGE_NAME ${PROJECT_NAME} CACHE STRING "Holoscan SDK")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Holoscan SDK"
    CACHE STRING "Package description for Holoscan SDK"
)
set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${CPACK_PACKAGE_NAME})

set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/nvidia/holoscan")

set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})

set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Julien Jomier")

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_SOURCE_DIR}/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_SOURCE_DIR}/README.md")

# Sets the package name as debian format
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

# Just one package will all the components
set(CPACK_DEB_COMPONENT_INSTALL 1)
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)

# List of the components that should be installed
set(CPACK_COMPONENTS_ALL
  holoscan # component from rapids-cmake export
  holoscan-core
  holoscan-gxf_extensions
  holoscan-gxf_libs
  holoscan-gxf_bins
  holoscan-modules
  holoscan-dependencies
  holoscan-examples
  holoscan-python_libs
  holoscan-cpack
  )


# - cuda-nvcc: needed to find Holoscan with CMake (FindCUDAToolkit requirement)
#   Note: not needed at runtime
# - cuda-cudart-dev: needed for holoscan core and some operators at build time and by Cupy at runtime
set(CPACK_DEBIAN_PACKAGE_DEPENDS
  "cuda-nvcc-12-6 | cuda-nvcc-12-9 | cuda-nvcc-12-8 | cuda-nvcc-12-7 | cuda-nvcc-12-5 | cuda-nvcc-12-4 | cuda-nvcc-12-3 | cuda-nvcc-12-2 |cuda-nvcc-12-1 | cuda-nvcc-12-0, \
  cuda-cudart-dev-12-6 | libcudart.so.12-dev"
)

# Recommended packages for core runtime functionality:
# - libnvinfer-bin: meta package including required nvinfer libs.
#   Needed for all inference backends
#   Note: only libnvinfer, libnvonnxparsers, libnvinfer-plugin needed at runtime
# - libcublas: needed by CuPy, libtorch, and OnnxRuntime
#   Note: also a dependency of the libnvinfer packages
# - cuda-nvrtc: libtorch & CuPy dependency
#   Note: also a dependency of cuda-nvcc
#   Note: should be able to use libnvrtc.so.12, but doesn't work as of Holoscan SDK 2.4
# - libcufft: needed by cupy and OnnxRuntime inference backend
# - libcurand: needed by libtorch and cupy
# - libcusolver: needed by cupy
# - libcusparse: needed by cupy
# - libnpp-dev: needed for format_converter and bayer_demosaic operators
#   Note: only libnpp (non dev) needed at runtime
# - libnvjitlink: needed by cupy
# - nccl2: needed by cupy and Torch
# - libgomp1: needed by cupy
# - libvulkan1: needed for holoviz operator
# - libegl1: needed for holoviz operator in headless mode
# - libv4l-0: needed for v4l2 operator
# - python3-cloudpickle: needed for python distributed applications
# - python3-pip: needed for holoscan CLI (packager, runner)
# - libnuma1: needed for holoscan::core on ARM64
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "\
libnvinfer-bin (>=10.3), \
libcublas-12-6 | libcublas.so.12, \
cudnn9-cuda-12-6 | libcudnn.so.9, \
cuda-nvrtc-12-6 | cuda-nvrtc-12-9 | cuda-nvrtc-12-8 | cuda-nvrtc-12-7 | cuda-nvrtc-12-5 | cuda-nvrtc-12-4 | cuda-nvrtc-12-3 | cuda-nvrtc-12-2 | cuda-nvrtc-12-1 | cuda-nvrtc-12-0, \
libcufft-12-6 | libcufft.so.11, \
libcurand-12-6 | libcurand.so.10, \
libcusolver-12-6 | libcusolver.so.11, \
libcusparse-12-6 | libcusparse.so.12, \
libnpp-dev-12-6 | libnpp.so.12-dev, \
libnvjitlink-12-6 | libnvjitlink.so.12, \
libnccl2 | libnccl.so.2, \
libgomp1, \
libvulkan1, \
libegl1, \
libv4l-0, \
python3-cloudpickle, \
python3-pip"
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "${CPACK_DEBIAN_PACKAGE_RECOMMENDS}, \
libnuma1")
endif()

# Packages for optional features:
# - libcupti: needed for Torch inference backend
# - libnvToolsExt: needed for Torch inference backend
# - libcudnn: needed for Torch and OnnxRuntime
# - libcusparselt: needed for Torch inference backend
# - libpng, libjpeg, libopenblas: needed for Torch inference backend.
# - libjpeg needed by v4l2 for mjpeg support
set(CPACK_DEBIAN_PACKAGE_SUGGESTS "\
cuda-cupti-12-6 | libcupti.so.12, \
cuda-nvtx-12-6 | libnvToolsExt.so.1, \
libcudnn9-cuda-12 | libcudnn.so.9, \
libcusparselt0 | libcusparselt.so.0, \
libpng16-16, \
libjpeg-turbo8, \
libopenblas0"
)

include(CPack)

# Add the components to packages
message(STATUS "Components to pack: ${CPACK_COMPONENTS_ALL}")
foreach(component IN LISTS CPACK_COMPONENTS_ALL)
  cpack_add_component(${component})
endforeach()
