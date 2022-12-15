# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")

# Sets the package name as debian format
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)

# Just one package will all the components
set(CPACK_DEB_COMPONENT_INSTALL 1)
set(CPACK_COMPONENTS_GROUPING ONE_PER_GROUP)
set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS 1)

# List of the components that should be installed
set(CPACK_COMPONENTS_ALL 
  holoscan-core
  holoscan-gxf_extensions
  holoscan-gxf_libs
  holoscan-gxf_bins
  holoscan-modules
  holoscan-dependencies
  holoscan-apps
  holoscan-data
  holoscan-python_libs
  )

set(CPACK_DEBIAN_DEV_PACKAGE_DEPENDS "cuda (>=11.6.1), cuda (<<12.0), libnvinfer-bin (>= 8.2.3), libnvinfer-plugin8 (>=8.2.3),libnvinfer8 (>= 8.2.3), libnvonnxparsers8 (>= 8.2.3)")
set(CPACK_COMPONENT_APPS_DEPENDS DEV)

include(CPack)

message(STATUS "Components to pack: ${CPACK_COMPONENTS_ALL}")

# Create the Dev package
cpack_add_component_group(DEV "Holoscan - Dev")
cpack_add_component(holoscan-core GROUP DEV)
cpack_add_component(holoscan-python_libs GROUP DEV)
cpack_add_component(holoscan-gxf_extensions GROUP DEV)
cpack_add_component(holoscan-gxf_libs GROUP DEV)
cpack_add_component(holoscan-gxf_bins GROUP DEV)
cpack_add_component(holoscan-modules GROUP DEV)
cpack_add_component(holoscan-dependencies GROUP DEV)

# Create the Apps package
cpack_add_component_group(APPS "Holoscan - Sample Applications")
cpack_add_component(holoscan-apps GROUP APPS)
cpack_add_component(holoscan-data GROUP APPS)

