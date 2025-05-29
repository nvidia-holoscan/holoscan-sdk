# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(HOLOSCAN_GXF_COMPONENTS
    # For Holoscan to use and distribute
    app
    core
    cuda
    gxe
    logger
    multimedia
    rmm
    sample # dependency of GXF::app
    serialization
    std
    ucx
)

find_package(GXF 5.0 CONFIG REQUIRED
    COMPONENTS ${HOLOSCAN_GXF_COMPONENTS}
)
message(STATUS "Found GXF: ${GXF_DIR}")

# Workaround: If the GXF distribution implicitly includes an HTTP target dependency
# for other libraries, add it to the list of imports.
# https://jirasw.nvidia.com/browse/NVG-3245
if(TARGET GXF::http)
    list(APPEND HOLOSCAN_GXF_COMPONENTS http)
endif()

# Copy shared libraries and their headers to the GXF build folder
# to be found alongside Holoscan GXF extensions.

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

# Copy the GXF binaries to the build folder so that executables can find them
set(HOLOSCAN_GXF_LIB_DIR "${CMAKE_BINARY_DIR}/${HOLOSCAN_INSTALL_LIB_DIR}")
set(HOLOSCAN_GXF_BIN_DIR "${CMAKE_BINARY_DIR}/bin")
foreach(component ${HOLOSCAN_GXF_COMPONENTS})
    # Find binary location
    get_target_property(GXF_${component}_LOCATION GXF::${component} IMPORTED_LOCATION)
    if(NOT GXF_${component}_LOCATION)
        message(DEBUG "No generic location found for GXF::${component}, checking per build types")
        foreach(build_type RELEASE RELWITHDEBINFO DEBUG)
            get_target_property(GXF_${component}_LOCATION GXF::${component} IMPORTED_LOCATION_${build_type})
            if(GXF_${component}_LOCATION)
                message(DEBUG "Found GXF::${component} (${build_type}) at ${GXF_${component}_LOCATION}")
                break()
            endif()
        endforeach()
    endif()
    if(NOT GXF_${component}_LOCATION)
        message(FATAL_ERROR "No imported location found for GXF::${component}")
    endif()

    # Copy to build directory
    if(NOT "${component}" STREQUAL "gxe")
        file(COPY "${GXF_${component}_LOCATION}"
            DESTINATION "${HOLOSCAN_GXF_LIB_DIR}"
            FILE_PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ WORLD_READ
        )
        get_filename_component(${component}_filename ${GXF_${component}_LOCATION} NAME)
        set(HOLOSCAN_GXF_${component}_LOCATION "${HOLOSCAN_GXF_LIB_DIR}/${${component}_filename}")
        set_target_properties(GXF::${component} PROPERTIES
            IMPORTED_LOCATION_${_build_type} ${HOLOSCAN_GXF_${component}_LOCATION}
            IMPORTED_LOCATION ${HOLOSCAN_GXF_${component}_LOCATION}
        )
    else()
        file(COPY "${GXF_${component}_LOCATION}"
            DESTINATION "${HOLOSCAN_GXF_BIN_DIR}"
            FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        )
        set(HOLOSCAN_GXE_LOCATION "${HOLOSCAN_GXF_BIN_DIR}/gxe")
        set_target_properties(GXF::gxe PROPERTIES
            IMPORTED_LOCATION_${_build_type} {HOLOSCAN_GXE_LOCATION}
            IMPORTED_LOCATION ${HOLOSCAN_GXE_LOCATION}
        )

        # Patch `gxe` executable RUNPATH to find required GXF libraries in the self-contained HSDK installation.
        # GXF libraries are entirely self-contained and do not require RPATH updates.
        find_program(PATCHELF_EXECUTABLE patchelf)
        if(PATCHELF_EXECUTABLE)
            execute_process(
                COMMAND "${PATCHELF_EXECUTABLE}"
                    "--set-rpath"
                    "\$ORIGIN:\$ORIGIN/../${HOLOSCAN_INSTALL_LIB_DIR}"
                    "${HOLOSCAN_GXE_LOCATION}"
            )
        else()
            message(WARNING "Failed to patch the GXE executable RUNPATH. Must set LD_LIBRARY_PATH to use the executable.")
        endif()
    endif()
endforeach()

# Find the GXF Python module path (optional, not in cmake-based build as of yet)
if(NOT GXF_ROOT)
    cmake_path(GET "${GXF_DIR}" PARENT_PATH PARENT_PATH PARENT_PATH GXF_ROOT)
endif()
find_path(GXF_PYTHON_MODULE_PATH
    NAMES
        core/__init__.py
        core/Gxf.py
    PATHS ${GXF_ROOT}/python/gxf
)

# Test that the GXF Python module is in PYTHONPATH
find_package(Python3 COMPONENTS Interpreter REQUIRED)
if(HOLOSCAN_REGISTER_GXF_EXTENSIONS)
    # GXF Python module is required for registering GXF extensions
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import os; import gxf; print(os.pathsep.join(gxf.__path__).strip())"
        RESULT_VARIABLE GXF_MODULE_FOUND
        OUTPUT_VARIABLE GXF_MODULE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(NOT GXF_MODULE_FOUND EQUAL 0)
        message(FATAL_ERROR "GXF Python module not found in PYTHONPATH")
    endif()
    if(NOT GXF_MODULE_DIR STREQUAL "${GXF_PYTHON_MODULE_PATH}")
        message(WARNING
            "Expected GXF Python module at ${GXF_PYTHON_MODULE_PATH} but found at ${GXF_MODULE_DIR}."
            " Do you need to update your PYTHONPATH?")
    endif()
endif()

# Set variables in parent scope for use throughout the Holoscan project
set(GXF_INCLUDE_DIR ${GXF_INCLUDE_DIR} PARENT_SCOPE)
set(GXF_PYTHON_MODULE_PATH ${GXF_PYTHON_MODULE_PATH} PARENT_SCOPE)
set(HOLOSCAN_GXF_LIB_DIR ${HOLOSCAN_GXF_LIB_DIR} PARENT_SCOPE)
set(HOLOSCAN_GXF_BIN_DIR ${HOLOSCAN_GXF_BIN_DIR} PARENT_SCOPE)
set(HOLOSCAN_GXE_LOCATION ${HOLOSCAN_GXE_LOCATION} PARENT_SCOPE)
set(HOLOSCAN_GXF_COMPONENTS ${HOLOSCAN_GXF_COMPONENTS} PARENT_SCOPE)
