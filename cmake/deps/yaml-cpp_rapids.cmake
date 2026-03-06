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

set(YAML_CPP_VERSION 0.8.0)
find_package(yaml-cpp ${YAML_CPP_VERSION} REQUIRED)

if(yaml-cpp_FOUND)
    if(NOT EXISTS ${YAML_CPP_INCLUDE_DIR})
        message(WARNING "yaml-cpp is marked FOUND but is missing YAML_CPP_INCLUDE_DIR")
    endif()
    if(NOT TARGET yaml-cpp::yaml-cpp)
        message(WARNING "yaml-cpp is marked FOUND but is missing yaml-cpp::yaml-cpp target")
    endif()

    # Alias without namespace to meet GXF::core 5.3 INTERFACE_LINK_LIBRARIES requirement
    if(NOT TARGET yaml-cpp)
        add_library(yaml-cpp ALIAS yaml-cpp::yaml-cpp)
    endif()

    # Pass yaml-cpp installation through to custom "3rdparty" paths in Holoscan SDK installation
    install(DIRECTORY ${YAML_CPP_INCLUDE_DIR}/yaml-cpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/3rdparty/yaml-cpp"
        COMPONENT "holoscan-dependencies"
    )

    # Manually install the yaml-cpp static library to the output library folder
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _cmake_build_type_upper)
    get_target_property(YAML_CPP_LIB_PATH yaml-cpp::yaml-cpp IMPORTED_LOCATION_${_cmake_build_type_upper})
    if(NOT EXISTS "${YAML_CPP_LIB_PATH}")
        get_target_property(YAML_CPP_LIB_PATH yaml-cpp::yaml-cpp IMPORTED_LOCATION_RELEASE)
    endif()
    if(NOT EXISTS "${YAML_CPP_LIB_PATH}")
        get_target_property(YAML_CPP_LIB_PATH yaml-cpp::yaml-cpp IMPORTED_LOCATION)
    endif()
    if(EXISTS "${YAML_CPP_LIB_PATH}")
        install(
            FILES "${YAML_CPP_LIB_PATH}"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            COMPONENT "holoscan-dependencies"
        )
    else()
        message(FATAL_ERROR "yaml-cpp static library not found at: ${YAML_CPP_LIB_PATH}")
    endif()

    # Write custom "yaml-cpp-config.cmake" to reflect the updated 3rdparty output location
    # in the "holoscan" installation
    include(CMakePackageConfigHelpers)
    set(YAML_CPP_CONFIG_OUT "${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp-config.cmake")
    configure_package_config_file(
        ${CMAKE_CURRENT_LIST_DIR}/configs/yaml-cpp-config.cmake.in
        ${YAML_CPP_CONFIG_OUT}
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/yaml-cpp"
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp-config-version.cmake"
        VERSION ${YAML_CPP_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES
            ${YAML_CPP_CONFIG_OUT}
            "${CMAKE_CURRENT_BINARY_DIR}/yaml-cpp-config-version.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/yaml-cpp"
        COMPONENT "holoscan-dependencies"
    )

endif()
