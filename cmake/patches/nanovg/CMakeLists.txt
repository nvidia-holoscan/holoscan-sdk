# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.22)
project(nanovg VERSION 1.0)

# Paths
include(GNUInstallDirs)
set(NANOVG_CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/nanovg)
set(NANOVG_INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR}/nanovg)
set(NANOVG_CMAKE_CONFIG ${CMAKE_CURRENT_BINARY_DIR}/nanovg-config.cmake)
set(NANOVG_CMAKE_CONFIG_VERSION ${CMAKE_CURRENT_BINARY_DIR}/nanovg-config-version.cmake)

# Create library
list(APPEND NANOVG_INCLUDE_FILES
  src/fontstash.h
  src/nanovg_gl.h
  src/nanovg_gl_utils.h
  src/nanovg.h
  src/stb_image.h
  src/stb_truetype.h
)
add_library(nanovg src/nanovg.c)
target_include_directories(nanovg
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Setup configuration files
include(CMakePackageConfigHelpers)
configure_package_config_file(nanovg-config.cmake.in
  ${NANOVG_CMAKE_CONFIG}
  INSTALL_DESTINATION ${NANOVG_CMAKE_INSTALL_DIR}
)
write_basic_package_version_file(
  ${NANOVG_CMAKE_CONFIG_VERSION}
  VERSION ${CMAKE_PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install
install(TARGETS nanovg
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  EXPORT nanovg-targets
)
install(EXPORT nanovg-targets
  DESTINATION ${NANOVG_CMAKE_INSTALL_DIR}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/nanovg-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/nanovg-config-version.cmake
  DESTINATION ${NANOVG_CMAKE_INSTALL_DIR}
)
install(FILES ${NANOVG_INCLUDE_FILES}
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
