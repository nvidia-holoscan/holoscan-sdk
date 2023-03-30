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

# Create GXF imported cmake targets
#
# This module defines GXF_FOUND if all GXF libraries are found or
# if the required libraries (COMPONENTS property in find_package)
# are found.
#
# A new imported target is created for each component (library)
# under the GXF namespace (GXF::${component_name})
#
# Note: this leverages the find-module paradigm [1]. The config-file paradigm [2]
# is recommended instead in CMake.
# [1] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#config-file-packages
# [2] https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html#find-module-packages

# Define environment
if(CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL amd64)
    set(_internal_GXF_recipe "gxf_x86_64")
    set(_public_GXF_recipe "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
    set(_internal_GXF_recipe "gxf_jetpack50")
    set(_public_GXF_recipe "arm64")
else()
    message(FATAL_ERROR "CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR} is not an architecture supported by GXF")
endif()

if(NOT HOLOSCAN_INSTALL_LIB_DIR)
    if(DEFINED HOLOSCAN_SDK_PATH)
        # Find library directory from HOLOSCAN_SDK_PATH
        find_path(HOLOSCAN_INSTALL_LIB_DIR
            NAMES libholoscan.so
            PATHS ${HOLOSCAN_SDK_PATH}/lib ${HOLOSCAN_SDK_PATH}/lib64
            NO_DEFAULT_PATH
            REQUIRED
        )

        # Take only file name from path
        get_filename_component(HOLOSCAN_INSTALL_LIB_DIR "${HOLOSCAN_INSTALL_LIB_DIR}" NAME)
    else()
        message(FATAL_ERROR "Unable to guess HOLOSCAN_INSTALL_LIB_DIR from HOLOSCAN_SDK_PATH")
    endif()
endif()

# Need PatchELF to update the RPATH of the libs
find_program(PATCHELF_EXECUTABLE patchelf)
if(NOT PATCHELF_EXECUTABLE)
    message(FATAL_ERROR "Please specify the PATCHELF executable")
endif()

# Library names
list(APPEND _GXF_EXTENSIONS
    behavior_tree
    cuda
    multimedia
    network
    npp
    python_codelet
    sample
    serialization
    std
    tensor_rt
    videoencoderio
    videoencoder
    videodecoderio
    videodecoder
)

# Common headers
find_path(GXF_common_INCLUDE_DIR
    NAMES common/
    REQUIRED
)
mark_as_advanced(GXF_common_INCLUDE_DIR)
list(APPEND GXF_INCLUDE_DIR_VARS GXF_common_INCLUDE_DIR)

# Libraries and their headers
list(APPEND _GXF_LIBRARIES ${_GXF_EXTENSIONS} core)

foreach(component IN LISTS _GXF_LIBRARIES)
    # headers
    find_path(GXF_${component}_INCLUDE_DIR
        NAMES "gxf/${component}/"
    )
    mark_as_advanced(GXF_${component}_INCLUDE_DIR)
    list(APPEND GXF_INCLUDE_DIR_VARS GXF_${component}_INCLUDE_DIR)

    # library
    find_library(GXF_${component}_LIBRARY
        NAMES "gxf_${component}"
        PATH_SUFFIXES
        "${_internal_GXF_recipe}/${component}"
        "${_public_GXF_recipe}/${component}"
    )
    mark_as_advanced(GXF_${component}_LIBRARY)
    list(APPEND GXF_LIBRARY_VARS GXF_${component}_LIBRARY)

    # create imported target
    if(GXF_${component}_LIBRARY AND GXF_${component}_INCLUDE_DIR)

        set(gxf_component_location "${GXF_${component}_LIBRARY}")

        if(NOT TARGET GXF::${component})
            # Assume SHARED, though technically UNKNOWN since we don't enforce .so
            add_library(GXF::${component} SHARED IMPORTED)

            # Copy a GXF library to ${CMAKE_BINARY_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}/ folder
            file(COPY "${GXF_${component}_LIBRARY}"
                DESTINATION "${CMAKE_BINARY_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}"
                FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
            )

            # Set the internal location to the binary directory
            # This is need for RPATH to work
            get_filename_component(gxf_component_name "${GXF_${component}_LIBRARY}" NAME)
            set(gxf_component_location "${CMAKE_BINARY_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}/${gxf_component_name}")

            # Install a GXF library as a component 'holoscan-gxf_libs'
            # Use the copied location since RPATH is changed
            install(FILES "${gxf_component_location}"
                DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}"
                COMPONENT "holoscan-gxf_libs"
            )
        endif()

        list(APPEND GXF_${component}_INCLUDE_DIRS
            ${GXF_${component}_INCLUDE_DIR}
            ${GXF_common_INCLUDE_DIR}
        )

        # Points to the copied location of GXF
        set_target_properties(GXF::${component} PROPERTIES
            IMPORTED_LOCATION "${gxf_component_location}"

            # Without this, make and ninja's behavior is different.
            # GXF's shared libraries doesn't seem to set soname.
            # (https://gitlab.kitware.com/cmake/cmake/-/issues/22307)
            IMPORTED_NO_SONAME ON
            INTERFACE_INCLUDE_DIRECTORIES "${GXF_${component}_INCLUDE_DIRS}"
        )

        list(APPEND _GXF_LIB_RPATH "\$ORIGIN" "\$ORIGIN/gxf_extensions")

        # The video encoder/decoder libraries need an extra path for aarch64
        # To find the right l4t libraries
        if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
            if(component STREQUAL videoencoderio
            OR component STREQUAL videoencoder
            OR component STREQUAL videodecoderio
            OR component STREQUAL videodecoder)
                list(APPEND _GXF_LIB_RPATH "/usr/lib/aarch64-linux-gnu/tegra/")
            endif()
        endif()

        # Patch RUNPATH
        list(JOIN _GXF_LIB_RPATH ":" _GXF_LIB_RPATH)
        execute_process(COMMAND
            "${PATCHELF_EXECUTABLE}"
            "--set-rpath"
            "${_GXF_LIB_RPATH}"
            "${gxf_component_location}"
        )
        unset(_GXF_LIB_RPATH)

        set(GXF_${component}_FOUND TRUE)
    else()
        set(GXF_${component}_FOUND FALSE)
    endif()
endforeach()

unset(_GXF_EXTENSIONS)
unset(_GXF_LIBRARIES)

# Find version
if(GXF_core_INCLUDE_DIR)
    # Note: "kGxfCoreVersion \"(.*)\"$" does not work with a simple string
    # REGEX (doesn't stop and EOL, neither $ nor \n), so we first extract
    # the line with file(STRINGS), then the version with string(REGEX)
    file(STRINGS "${GXF_core_INCLUDE_DIR}/gxf/core/gxf.h" _GXF_VERSION_LINE
        REGEX "kGxfCoreVersion"
    )
    string(REGEX MATCH "kGxfCoreVersion \"(.*)\"" _ ${_GXF_VERSION_LINE})
    set(GXF_VERSION ${CMAKE_MATCH_1})
    unset(_GXF_VERSION_LINE)
endif()

# Install the Deepstream dependencies only on x86_64
if(_public_GXF_recipe STREQUAL x86_64)
  file(GLOB DS_DEPS_NVUTILS "${GXF_core_INCLUDE_DIR}/x86_64/nvutils/*.so")
  file(GLOB DS_DEPS_CUVIDV4L2 "${GXF_core_INCLUDE_DIR}/x86_64/cuvidv4l2/*.so")

  foreach(dsdep IN LISTS DS_DEPS_NVUTILS DS_DEPS_CUVIDV4L2)

    # Patch RUNPATH
    execute_process(COMMAND
    "${PATCHELF_EXECUTABLE}"
    "--set-rpath"
    "\$ORIGIN"
    "${dsdep}"
    )

    # Install the files
    install(FILES "${dsdep}"
            DESTINATION "${HOLOSCAN_INSTALL_LIB_DIR}"
            COMPONENT "holoscan-gxf_libs"
            )
  endforeach()

  # Create a symlink
  INSTALL(CODE "execute_process( \
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    libnvv4l2.so \
    libv4l2.so.0 \
    COMMAND ${CMAKE_COMMAND} -E create_symlink \
    libnvv4lconvert.so \
    libv4lconvert.so.0 \
    WORKING_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/${HOLOSCAN_INSTALL_LIB_DIR}\"
    )"
    COMPONENT "holoscan-gxf_libs"
  )
endif()


# GXE
find_program(GXF_gxe_PATH
    NAMES gxe
    PATH_SUFFIXES
    "${_internal_GXF_recipe}/gxe"
    "${_public_GXF_recipe}/gxe"
)

if(GXF_gxe_PATH)
    if(NOT TARGET GXF::gxe)
        add_executable(GXF::gxe IMPORTED)
    endif()

    # 'gxe' is read-only file, so we need to update the permissions before patching
    # with patchelf (https://github.com/NixOS/nixpkgs/issues/14440).
    # Note that 'file(CHMOD)' is supported since later CMake versions(>=3.19)
    file(CHMOD ${GXF_gxe_PATH}
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )

    # Patch RUNPATH so that it can find libgxf_core.so library.
    execute_process(COMMAND
        "${PATCHELF_EXECUTABLE}"
        "--set-rpath"
        "\$ORIGIN:\$ORIGIN/../${HOLOSCAN_INSTALL_LIB_DIR}"
        "${GXF_gxe_PATH}"
    )

    set_target_properties(GXF::gxe PROPERTIES
        IMPORTED_LOCATION "${GXF_gxe_PATH}"
    )
    set(GXF_gxe_FOUND TRUE)
else()
    set(GXF_gxe_FOUND FALSE)
endif()

# Generate GXF_FOUND
include(FindPackageHandleStandardArgs)

if(GXF_FIND_COMPONENTS)
    # ... based on requested components/libraries
    find_package_handle_standard_args(GXF
        FOUND_VAR GXF_FOUND
        VERSION_VAR GXF_VERSION
        HANDLE_COMPONENTS # Looks for GXF_${component}_FOUND
    )
else()
    # ... need all the libraries
    find_package_handle_standard_args(GXF
        FOUND_VAR GXF_FOUND
        VERSION_VAR GXF_VERSION
        REQUIRED_VARS ${GXF_INCLUDE_DIR_VARS} ${GXF_LIBRARY_VARS} GXF_gxe_PATH
    )
endif()

# Clean
unset(_internal_GXF_recipe)
unset(_public_GXF_recipe)
