# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy NOTICE file for installation
install(FILES ${CMAKE_CURRENT_LIST_DIR}/cpack/NOTICE.txt
    DESTINATION .
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

# Method to get all CUDA debian package names up to the current MAJOR-MINOR version
if(NOT DEFINED CUDAToolkit_VERSION)
  find_package(CUDAToolkit REQUIRED)
endif()
function(get_minor_version_packages dep_name output_string)
  foreach(minor RANGE ${CUDAToolkit_VERSION_MINOR} 0 -1)
    list(APPEND packages "${dep_name}-${CUDAToolkit_VERSION_MAJOR}-${minor}")
  endforeach()
  list(JOIN packages " | " ${output_string})
  set(${output_string} ${${output_string}} PARENT_SCOPE)
endfunction()
set(CUDA_MAJOR ${CUDAToolkit_VERSION_MAJOR})

# - cuda-nvcc: needed to find Holoscan with CMake (FindCUDAToolkit requirement)
#   Note: not needed at runtime
# - cuda-cudart-dev: needed for holoscan core and some operators
#   Note: only cuda-cudart (non dev) needed at runtime
get_minor_version_packages("cuda-nvcc" NVCC_PACKAGES)
get_minor_version_packages("cuda-cudart-dev" CUDART_DEV_PACKAGES)
set(CPACK_DEBIAN_PACKAGE_DEPENDS
"${NVCC_PACKAGES}, libcudart.so.${CUDA_MAJOR}-dev | ${CUDART_DEV_PACKAGES}"
)
# - libnvinfer-bin: meta package including required nvinfer libs, cublas, and cudnn.
#   Needed for TensorRT and OnnxRuntime inference backends
# - cuda-nvrtc: needed by cupy
#   Note: only libcudart (non dev) needed at runtime
#   Note: also a dependency of cuda-nvcc
# - libcublas: needed by TensorRT and OnnxRuntime inference backends
#   Note: also a dependency of the libnvinfer packages
# - libcufft: needed by cupy and OnnxRuntime inference backend
# - libcurand: needed by cupy
# - libcusolver: needed by cupy
# - libcusparse: needed by cupy
# - libnpp-dev: needed for format_converter and bayer_demosaic operators
#   Note: only libnpp (non dev) needed at runtime
# - libnvjitlink: needed by cupy
# - libgomb1: needed by cupy
# - libvulkan1, libx...: needed for holoviz operator
# - libegl1: needed for holoviz operator in headless mode
# - libv4l-0: needed for v4l2 operator
# - python3-cloudpickle: needed for python distributed applications
# - python3-pip: needed for holoscan CLI (packager, runner)
get_minor_version_packages("cuda-nvrtc" NVRTC_PACKAGES)
get_minor_version_packages("libcublas" CUBLAS_PACKAGES)
get_minor_version_packages("libcufft" CUFFT_PACKAGES)
get_minor_version_packages("libcurand" CURAND_PACKAGES)
get_minor_version_packages("libcusolver" CUSOLVER_PACKAGES)
get_minor_version_packages("libcusparse" CUSPARSE_PACKAGES)
get_minor_version_packages("libnpp-dev" NPP_DEV_PACKAGES)
get_minor_version_packages("libnvjitlink" NVJITLINK_PACKAGES)
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS
"libnvinfer-bin (>= 8.6.1), \
${NVRTC_PACKAGES}, \
libcublas.so.${CUDA_MAJOR} | ${CUBLAS_PACKAGES}, \
libcufft.so.${CUDA_MAJOR} | ${CUFFT_PACKAGES}, \
libcurand.so.${CUDA_MAJOR} | ${CURAND_PACKAGES}, \
libcusolver.so.${CUDA_MAJOR} | ${CUSOLVER_PACKAGES}, \
libcusparse.so.${CUDA_MAJOR} | ${CUSPARSE_PACKAGES}, \
libnpp.so.${CUDA_MAJOR}-dev | ${NPP_DEV_PACKAGES}, \
libnvJitLink.so.${CUDA_MAJOR} | ${NVJITLINK_PACKAGES}, \
libgomb1, \
libvulkan1, libx11-6, libxcb-glx0, libxcb-glx0, libxcursor1, libxi6, libxinerama1, libxrandr2, \
libegl1, \
libv4l-0, \
python3-cloudpickle, \
python3-pip"
)
# - libpng, libjpeg, libopenblas: needed for Torch inference backend
set(CPACK_DEBIAN_PACKAGE_SUGGESTS "libpng16-16, libjpeg-turbo8, libopenblas0")

include(CPack)

# Add the components to packages
message(STATUS "Components to pack: ${CPACK_COMPONENTS_ALL}")
foreach(component IN LISTS CPACK_COMPONENTS_ALL)
  cpack_add_component(${component})
endforeach()
