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

# Set CMAKE_PREFIX_PATH to locate GXF
if(NOT GXF_SDK_PATH)
    set(GXF_SDK_PATH "${CMAKE_SOURCE_DIR}/.cache/gxf")
    list(APPEND CMAKE_PREFIX_PATH "${GXF_SDK_PATH}")
    message("GXF_SDK_PATH is not set. Using '${GXF_SDK_PATH}'")

    set(GXF_SDK_PATH "${GXF_SDK_PATH}" PARENT_SCOPE)
else()
    list(APPEND CMAKE_PREFIX_PATH "${GXF_SDK_PATH}")
    message("GXF_SDK_PATH is set to ${GXF_SDK_PATH}")
endif()

find_package(GXF 2.4 MODULE REQUIRED
    COMPONENTS
    core
    cuda
    gxe
    multimedia
    serialization
    std
)

# Utility to get a folder path for GXF binaries
function(get_gxf_binary_path location)
    # Find a folder path for GXF binaries
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL amd64)
        find_path(_GXF_folder
            NAMES core/libgxf_core.so
            PATHS "${GXF_SDK_PATH}/gxf_x86_64" "${GXF_SDK_PATH}/x86_64"
            NO_DEFAULT_PATH
            REQUIRED
        )
    elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
        find_path(_GXF_folder
            NAMES core/libgxf_core.so
            PATHS "${GXF_SDK_PATH}/gxf_jetpack50" "${GXF_SDK_PATH}/arm64"
            NO_DEFAULT_PATH
            REQUIRED
        )
    else()
        message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} is not an architecture supported by GXF")
    endif()

    set(${location} ${_GXF_folder} PARENT_SCOPE)
endfunction()

get_gxf_binary_path(_GXF_folder)

# Copy gxe binary to the current binary directory
file(COPY "${_GXF_folder}/gxe/gxe"
    DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/bin"
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install gxe binary to the binary directory of the CMAKE_INSTALL_PREFIX
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/bin/gxe"
    DESTINATION "bin"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    COMPONENT "holoscan-gxf_bins"
)

# Clean
unset(_GXF_folder)
