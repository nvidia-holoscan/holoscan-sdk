# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

# Retrieves the magic_enum header-only library.
# For the purpose of CMake export handling, we want to treat magic_enum as an
# IMPORTED interface library so that it does not appear as a target in the
# Holoscan SDK CMake export set.
# To accomplish we download the library, skip its default CMake configuration,
# and instead define the custom IMPORTED target and export configuration.
set(magic_enum_VERSION 0.9.3)
rapids_cpm_find(magic_enum ${magic_enum_VERSION}
    CPM_ARGS
        GITHUB_REPOSITORY Neargye/magic_enum
        GIT_TAG v${magic_enum_VERSION}
        GIT_SHALLOW TRUE
        DOWNLOAD_ONLY TRUE
        EXCLUDE_FROM_ALL
)

# Set 'magic_enum_SOURCE_DIR' with PARENT_SCOPE so that
# root project can use it to include headers
set(magic_enum_SOURCE_DIR ${magic_enum_SOURCE_DIR} PARENT_SCOPE)

if(magic_enum_ADDED)
    # Define the custom IMPORTED target to avoid adding magic_enum
    # to the Holoscan SDK CMake export set.
    if(NOT magic_enum_SOURCE_DIR)
        message(FATAL_ERROR "magic_enum_SOURCE_DIR is not set")
    endif()
    if(NOT EXISTS "${magic_enum_SOURCE_DIR}/include")
        message(FATAL_ERROR "magic_enum include directory not found at ${magic_enum_SOURCE_DIR}/include")
    endif()
    add_library(magic_enum::magic_enum INTERFACE IMPORTED)
    target_include_directories(magic_enum::magic_enum
        INTERFACE ${magic_enum_SOURCE_DIR}/include
    )

    # Install the headers needed for development with the SDK
    include(GNUInstallDirs)
    install(FILES ${magic_enum_SOURCE_DIR}/include/magic_enum.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/magic_enum"
        COMPONENT "holoscan-dependencies"
    )

    # Write custom "magic_enum-config.cmake" to reflect the updated 3rdparty output location
    # in the "holoscan" installation
    include(CMakePackageConfigHelpers)
    set(MAGIC_ENUM_CONFIG_OUT "${CMAKE_CURRENT_BINARY_DIR}/magic_enum-config.cmake")
    configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/configs/magic_enum-config.cmake.in
        ${MAGIC_ENUM_CONFIG_OUT}
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/magic_enum"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/magic_enum-config-version.cmake"
        VERSION ${magic_enum_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES
            ${MAGIC_ENUM_CONFIG_OUT}
            "${CMAKE_CURRENT_BINARY_DIR}/magic_enum-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/magic_enum"
        COMPONENT "holoscan-dependencies"
    )
endif()
