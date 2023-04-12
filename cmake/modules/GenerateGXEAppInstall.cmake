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

# Generates a bash script that runs GXE with the provided shared libraries
#
# GXF::gxe is an imported target that wasn't configured by CMake. To enable
# running it and providing plugins, we can write a command in a bash script.
#
# The following macro will generate a bash script which:
# - is named after the NAME property
# - passes the YAML property to the --app argument
# - generates a manifest yaml file based on the EXTENSIONS and
# passes it to the --manifest argument
# - adds GXF::core to the LD_LIBRARY_PATH env variable to resolve runtime
# linking
# Utility to get a library file path

function(create_gxe_application)
    # Parse arguments
    list(APPEND GXE_APP_OPTION_VARS "")
    list(APPEND GXE_APP_SINGLE_VALUE_VARS "NAME" "YAML" "COMPONENT")
    list(APPEND GXE_APP_MULTI_VALUE_VARS "EXTENSIONS")
    cmake_parse_arguments(
        GXE_APP
        "${GXE_APP_OPTION_VARS}"
        "${GXE_APP_SINGLE_VALUE_VARS}"
        "${GXE_APP_MULTI_VALUE_VARS}"
        ${ARGN}
    )

    # Ensure we have GXF::gxe
    if(NOT TARGET GXF::gxe)
        message(FATAL_ERROR "Can't create application without GXF::gxe. Please add gxe to the list of COMPONENTS in find_package(GXF)")
    endif()

    if(NOT TARGET GXF::core)
        message(FATAL_ERROR "Can't create application without GXF::core. Please add core to the list of COMPONENTS in find_package(GXF)")
    endif()

    # Ensure we have the app YAML
    get_filename_component(GXE_APP_YAML ${GXE_APP_YAML} REALPATH)

    if(NOT EXISTS ${GXE_APP_YAML})
        message(FATAL_ERROR "Can't create a GXE application without an app yaml.")
    endif()

    # Ensure we have a NAME
    if(NOT DEFINED GXE_APP_NAME)
        message(FATAL_ERROR "create_gxe_application requires a NAME argument")
    endif()

    # If 'COMPONENT' option is not defined
    if(NOT DEFINED GXE_APP_COMPONENT)
        set(GXE_APP_COMPONENT "holoscan-apps")
    endif()

    # Copy the yaml file (at configure time)
    get_filename_component(GXE_APP_YAML_NAME ${GXE_APP_YAML} NAME)
    file(COPY ${GXE_APP_YAML} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

    # Create manifest file
    list(APPEND GXE_APP_MANIFEST_CONTENT "extensions:")

    if(DEFINED GXE_APP_EXTENSIONS)
        foreach(extension IN LISTS GXE_APP_EXTENSIONS)
            # If the target exists
            if(TARGET ${extension})
              get_target_property(lib ${extension} IMPORTED_LOCATION)

              # If not lib we use the location of the target
              if(NOT lib)
                set(lib "$<TARGET_FILE:${extension}>")
              endif()
            elseif(GXF_EXTENSIONS_DIR) # use the full path and check if the target exists at generation time
              set(lib "$<$<TARGET_EXISTS:${extension}>:$<TARGET_FILE:${extension}>>$<$<NOT:$<TARGET_EXISTS:${extension}>>:${GXF_EXTENSIONS_DIR}/lib${extension}.so>")
            else()
              message(FATAL_ERROR "Target not found for ${extension} and GXF_EXTENSTIONS_DIR not set")
            endif()
            list(APPEND GXE_APP_MANIFEST_CONTENT ${lib})
            unset(lib)
        endforeach()
    endif()

    list(JOIN GXE_APP_MANIFEST_CONTENT "\n- " GXE_APP_MANIFEST_CONTENT)
    set(GXE_APP_MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_NAME}_manifest.yaml)
    file(GENERATE
        OUTPUT ${GXE_APP_MANIFEST}
        CONTENT ${GXE_APP_MANIFEST_CONTENT}
    )

    # Get the location of GXE
    get_target_property(gxelocation GXF::gxe IMPORTED_LOCATION)

    # Create bash script
    set(GXE_APP_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_NAME}")
    file(GENERATE
        OUTPUT "${GXE_APP_EXECUTABLE}"
        CONTENT
        "#!/usr/bin/env bash
export LD_LIBRARY_PATH=$(pwd):${GXF_LIB_DIR}:${GXF_EXTENSIONS_DIR}:\${LD_LIBRARY_PATH}
${gxelocation} --app ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_YAML_NAME} --manifest ${GXE_APP_MANIFEST} $@
"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
endfunction()
