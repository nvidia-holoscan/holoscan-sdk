# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(FetchContent)

set(EIGEN_PATCH_FILEPATH "${CMAKE_SOURCE_DIR}/cmake/deps/patches/eigen3_urm_neon_memcpy_fix.patch")

FetchContent_Declare(
    eigen3
    URL https://edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/eigen/eigen-3.4.0.tar.gz
    # sha256sum eigen-3.4.0.tar.gz
    URL_HASH SHA256=e4b6347ba6e9874f59de6f6f972e48f11c92b08331497152b7d99fd00b9e9aee
)

FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3)
    # eigen3_SOURCE_DIR is now available
endif()

# --------------------------------------------------------------------
# Apply the NEON memcpy-fix patch
# (Addresses GCC's -Werror=class-memaccess, which triggers an error on arm64)
# --------------------------------------------------------------------

# Locate the "patch" program
find_program(PATCH_EXECUTABLE patch)
if(NOT PATCH_EXECUTABLE)
    message(FATAL_ERROR "Cannot find the 'patch' tool needed to patch Eigen.")
endif()

# Apply the patch once (idempotent thanks to -N)
execute_process(
    COMMAND "${PATCH_EXECUTABLE}" -N -p1 -i "${EIGEN_PATCH_FILEPATH}"
    WORKING_DIRECTORY "${eigen3_SOURCE_DIR}"
    RESULT_VARIABLE _eigen_patch_result
)

if(_eigen_patch_result GREATER 1)   # 0 = success, 1 = already applied
    message(FATAL_ERROR "Failed to apply Eigen NEON memcpy patch "
                        "(exit code ${_eigen_patch_result}).")
endif()

# Create a regular INTERFACE library for Eigen's properties
add_library(eigen3_interface INTERFACE)

# Define include directory for the interface library
# Use generator expressions to handle both build-time and install-time include paths
target_include_directories(eigen3_interface INTERFACE
    $<BUILD_INTERFACE:${eigen3_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include/3rdparty>
)

# Set EIGEN_MPL2_ONLY flag to restrict Eigen usage to MPL2-licensed code only
target_compile_definitions(eigen3_interface INTERFACE
    EIGEN_MPL2_ONLY
)

# Create an ALIAS target `holoscan::eigen3`
add_library(holoscan::eigen3 ALIAS eigen3_interface)

# Set the EXPORT_NAME property on the original interface target
# This name is used when the target is exported as part of a CMake package.
set_target_properties(eigen3_interface PROPERTIES
    EXPORT_NAME eigen3  # This will allow it to be found as Holoscan::eigen3
)

# Install the Eigen headers for SDK development.
# This makes the headers available in the CMAKE_INSTALL_PREFIX.
if(IS_DIRECTORY "${eigen3_SOURCE_DIR}/include/Eigen")
    install(
        DIRECTORY "${eigen3_SOURCE_DIR}/include/Eigen/" # Source is the 'Eigen' folder itself
        DESTINATION include/3rdparty/Eigen                             # Destination path relative to CMAKE_INSTALL_PREFIX
        COMPONENT "holoscan-dependencies"                              # Matches existing component name
    )
else()
    message(FATAL_ERROR "Eigen headers not found in ${eigen3_SOURCE_DIR}/include/Eigen. "
                        "The downloaded Eigen archive from URM might have an unexpected structure, or the download failed.")
endif()
