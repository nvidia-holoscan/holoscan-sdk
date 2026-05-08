# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_program(PATCH_EXECUTABLE patch REQUIRED)
set(EIGEN_PATCH_FILEPATH "${CMAKE_SOURCE_DIR}/cmake/deps/patches/eigen3_urm_neon_memcpy_fix.patch")

# Use the minimized Eigen 3.4.0 source distribution from edge.urm.nvidia.com,
# which has been reduced to just Eigen header files (no utilities, tests, CMake, etc)
include(FetchContent)
FetchContent_Declare(
    eigen3
    URL https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/eigen/eigen-3.4.0.tar.gz
    # sha256sum eigen-3.4.0.tar.gz
    URL_HASH SHA256=e4b6347ba6e9874f59de6f6f972e48f11c92b08331497152b7d99fd00b9e9aee

    # --------------------------------------------------------------------
    # Apply the NEON memcpy-fix patch
    # (Addresses GCC's -Werror=class-memaccess, which triggers an error on arm64)
    # --------------------------------------------------------------------
    UPDATE_DISCONNECTED TRUE
    PATCH_COMMAND "${PATCH_EXECUTABLE}" -N -p1 -i "${EIGEN_PATCH_FILEPATH}"
)
FetchContent_MakeAvailable(eigen3)

# Use FindEigen3.cmake from Eigen project:
# https://gitlab.com/libeigen/eigen/-/blob/3.4.1/cmake/FindEigen3.cmake
# CMake rules have been stripped from the Eigen 3.4 source package on edge.urm.nvidia.com,
# so we instead use the FindEigen3.cmake module to define the Eigen3 import strategy.
set(Eigen3_ROOT "${eigen3_SOURCE_DIR}/include")
find_package(Eigen3 3.4 REQUIRED MODULE)

# Set EIGEN_MPL2_ONLY flag to restrict Eigen usage to MPL2-licensed code only
target_compile_definitions(Eigen3::Eigen INTERFACE
    EIGEN_MPL2_ONLY
)

# Create an ALIAS target `holoscan::eigen3` for backwards compatibility
add_library(holoscan::eigen3 ALIAS Eigen3::Eigen)

# Install the Eigen headers and import rules for SDK development.
# This makes the headers available in the CMAKE_INSTALL_PREFIX.
install(
    DIRECTORY "${eigen3_SOURCE_DIR}/include" # Source is the 'Eigen' folder itself
    DESTINATION include/3rdparty/Eigen                             # Destination path relative to CMAKE_INSTALL_PREFIX
    COMPONENT "holoscan-dependencies"                              # Matches existing component name
)
install(
    FILES ${holoscan_SOURCE_DIR}/cmake/modules/FindEigen3.cmake
    DESTINATION ${HOLOSCAN_INSTALL_LIB_DIR}/cmake/holoscan
    COMPONENT "holoscan-dependencies"
)
