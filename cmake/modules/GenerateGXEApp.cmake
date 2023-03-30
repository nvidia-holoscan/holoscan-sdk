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
function(_get_lib_file_path location target)
    get_target_property(imported ${target} IMPORTED)

    if(imported)
        get_target_property(lib ${target} IMPORTED_LOCATION)

        # If the path starts with GXF folder, take only the file name
        get_gxf_binary_path(_GXF_folder)
        cmake_path(IS_PREFIX _GXF_folder "${lib}" NORMALIZE _match_result)

        if(_match_result)
            get_filename_component(lib "${lib}" NAME)
        endif()
    else()
        set(lib $<TARGET_FILE:${target}>)
    endif()

    set(${location} ${lib} PARENT_SCOPE)
endfunction()

# Utility to get a library file directory
function(_get_lib_file_dir location target)
    get_target_property(imported ${target} IMPORTED)

    if(imported)
        get_target_property(lib ${target} IMPORTED_LOCATION)
        get_filename_component(dir ${lib} DIRECTORY)
    else()
        set(dir $<TARGET_FILE_DIR:${target}>)
    endif()

    set(${location} ${dir} PARENT_SCOPE)
endfunction()

# Utility to get all dependency library dirs for a target
function(_get_target_dependency_dirs dep_dirs target)
    if(TARGET ${target})
        # Check for target dependencies
        get_target_property(deps ${target} INTERFACE_LINK_LIBRARIES)

        if(deps)
            foreach(dep IN LISTS deps)
                _get_target_dependency_dirs(${dep_dirs} ${dep})
            endforeach()
        endif()

        # ... and itself
        _get_lib_file_dir(dir ${target})
    else()
        # Not a target: take the dir from the path instead
        get_filename_component(dir ${target} DIRECTORY)
    endif()

    if(NOT ${dir} MATCHES "^/usr*")
        # Only append non system libraries
        list(APPEND ${dep_dirs} ${dir})
    endif()

    set(${dep_dirs} ${${dep_dirs}} PARENT_SCOPE)
endfunction()

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

    # Get relative folder path for the app
    file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

    # Copy the yaml file
    get_filename_component(GXE_APP_YAML_NAME ${GXE_APP_YAML} NAME)
    cmake_path(APPEND GXE_APP_YAML_RELATIVE_PATH ${app_relative_dest_path} ${GXE_APP_YAML_NAME})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_YAML_NAME}
        COMMAND ${CMAKE_COMMAND} -E copy ${GXE_APP_YAML} ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_YAML_NAME}
        DEPENDS ${GXE_APP_YAML}
    )
    install(FILES ${GXE_APP_YAML}
        DESTINATION ${app_relative_dest_path}
        COMPONENT "${GXE_APP_COMPONENT}"
    )

    # Create manifest file
    list(APPEND GXE_APP_MANIFEST_CONTENT "extensions:")

    if(DEFINED GXE_APP_EXTENSIONS)
        foreach(extension IN LISTS GXE_APP_EXTENSIONS)
            _get_lib_file_path(lib ${extension})
            list(APPEND GXE_APP_MANIFEST_CONTENT ${lib})
            unset(lib)
        endforeach()
    endif()

    list(JOIN GXE_APP_MANIFEST_CONTENT "\n- " GXE_APP_MANIFEST_CONTENT)
    set(GXE_APP_MANIFEST_ORIGINAL ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_NAME}_manifest_original.yaml)
    file(GENERATE
        OUTPUT ${GXE_APP_MANIFEST_ORIGINAL}
        CONTENT ${GXE_APP_MANIFEST_CONTENT}
    )

    # Create a manifest file with the file paths replaced to relative paths
    set(GXE_APP_MANIFEST ${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_NAME}_manifest.yaml)
    set(GXE_APP_MANIFEST_RELATIVE_PATH ${app_relative_dest_path}/${GXE_APP_NAME}_manifest.yaml)
    get_gxf_binary_path(_GXF_folder)
    add_custom_command(OUTPUT "${GXE_APP_MANIFEST}"
        COMMAND sed
        -e "s|${_GXF_folder}/|./gxf/|g" # '/workspace/holoscan-sdk/.cache/gxf/gxf_x86_64/' => './gxf/'
        -e "s|${CMAKE_BINARY_DIR}/|./|g" # '/workspace/holoscan-sdk/build/' => './'
        ${GXE_APP_MANIFEST_ORIGINAL} > ${GXE_APP_MANIFEST}
        VERBATIM
        DEPENDS "${GXE_APP_MANIFEST_ORIGINAL}"
    )
    install(FILES ${GXE_APP_MANIFEST}
        DESTINATION ${app_relative_dest_path}
        COMPONENT "${GXE_APP_COMPONENT}"
    )

    # Create bash script
    set(GXE_APP_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_NAME}")
    file(GENERATE
        OUTPUT "${GXE_APP_EXECUTABLE}"
        CONTENT
        "#!/usr/bin/env bash
export LD_LIBRARY_PATH=$(pwd):$(pwd)/${HOLOSCAN_INSTALL_LIB_DIR}:\${LD_LIBRARY_PATH}
./bin/gxe --app ${GXE_APP_YAML_RELATIVE_PATH} --manifest ${GXE_APP_MANIFEST_RELATIVE_PATH} $@
"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )
    install(FILES ${GXE_APP_EXECUTABLE}
        DESTINATION ${app_relative_dest_path}
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
        COMPONENT "${GXE_APP_COMPONENT}"
    )

    # Create custom targets for the generated files
    add_custom_target(${GXE_APP_NAME} ALL
        DEPENDS
            "${CMAKE_CURRENT_BINARY_DIR}/${GXE_APP_YAML_NAME}"
            "${GXE_APP_MANIFEST}"
            # TODO: fix name conflict dependency with `file(GENERATE)`
            # Would need to find a way to only generate at build time?
            # "${GXE_APP_EXECUTABLE}"
    )
endfunction()

# Install target to a sub directory that has the same folder structure as the source
function(install_gxf_extension target_name)
    cmake_parse_arguments(PARSE_ARGV 1 INSTALL_GXF_EXTENSION "" "COMPONENT" "")

    # If 'COMPONENT' option is not defined
    if(NOT DEFINED INSTALL_GXF_EXTENSION_COMPONENT)
        set(INSTALL_GXF_EXTENSION_COMPONENT "holoscan-gxf_extensions")
    endif()

    install(TARGETS "${target_name}"
        DESTINATION lib/gxf_extensions
        COMPONENT "${INSTALL_GXF_EXTENSION_COMPONENT}"
    )

    # If dependent '_lib' target is defined, add it to the install target
    if(TARGET "${target_name}_lib")
        install(TARGETS "${target_name}_lib"
            DESTINATION lib/gxf_extensions
            COMPONENT "${INSTALL_GXF_EXTENSION_COMPONENT}"
        )
    endif()
endfunction()