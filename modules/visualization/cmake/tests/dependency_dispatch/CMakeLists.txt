# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)
project(holoviz_dependency_dispatch_test LANGUAGES CXX)

foreach(_required_variable IN ITEMS
    HOLOSCAN_CMAKE_DIR
    HOLOVIZ_CMAKE_DIR
    HOLOVIZ_TEST_CACHE_DIR
    HOLOVIZ_TEST_MODE
    HOLOVIZ_TEST_SLANG_MODE)
  if(NOT DEFINED ${_required_variable})
    message(FATAL_ERROR "${_required_variable} is required")
  endif()
endforeach()
unset(_required_variable)

include(GNUInstallDirs)

# Exercise the examples-only stb dispatch as part of the contract.
set(HOLOSCAN_BUILD_EXAMPLES ON)

function(add_test_interface_target target_name alias_name)
  add_library(${target_name} INTERFACE)
  add_library(${alias_name} ALIAS ${target_name})
endfunction()

add_test_interface_target(holoviz_test_cuda_driver CUDA::cuda_driver)

function(write_test_slang_package prefix provider)
  set(_slang_config_dir "${prefix}/lib/cmake/slang")
  file(MAKE_DIRECTORY "${_slang_config_dir}")
  file(WRITE "${_slang_config_dir}/slangConfig.cmake"
    "set_property(GLOBAL PROPERTY HOLOVIZ_TEST_SLANG_PROVIDER ${provider})\n"
    "if(NOT TARGET slang::slang)\n"
    "  add_library(slang::slang INTERFACE IMPORTED)\n"
    "endif()\n")
  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${_slang_config_dir}/slangConfigVersion.cmake"
    VERSION 2026.14
    COMPATIBILITY SameMajorVersion)
endfunction()

if(HOLOVIZ_TEST_SLANG_MODE STREQUAL "auto" OR
    HOLOVIZ_TEST_SLANG_MODE STREQUAL "prebuilt" OR
    HOLOVIZ_TEST_SLANG_MODE STREQUAL "source")
  string(TOUPPER "${HOLOVIZ_TEST_SLANG_MODE}" HOLOVIZ_SLANG_PROVIDER)
  set(_stale_slang_root "${HOLOVIZ_TEST_CACHE_DIR}/stale")
  write_test_slang_package("${_stale_slang_root}" stale)
  set(slang_DIR "${_stale_slang_root}/lib/cmake/slang" CACHE PATH "" FORCE)
elseif(HOLOVIZ_TEST_SLANG_MODE STREQUAL "external_root")
  set(_external_slang_root "${HOLOVIZ_TEST_CACHE_DIR}/external")
  set(_stale_slang_root "${HOLOVIZ_TEST_CACHE_DIR}/stale")
  write_test_slang_package("${_external_slang_root}" external_root)
  write_test_slang_package("${_stale_slang_root}" stale)
  set(HOLOVIZ_SLANG_PROVIDER EXTERNAL)
  set(slang_ROOT "${_external_slang_root}" CACHE PATH "" FORCE)
  set(slang_DIR "${_stale_slang_root}/lib/cmake/slang" CACHE PATH "" FORCE)
elseif(HOLOVIZ_TEST_SLANG_MODE STREQUAL "external_dir")
  set(_external_slang_root "${HOLOVIZ_TEST_CACHE_DIR}/external")
  write_test_slang_package("${_external_slang_root}" external_dir)
  set(HOLOVIZ_SLANG_PROVIDER EXTERNAL)
  set(slang_DIR "${_external_slang_root}/lib/cmake/slang" CACHE PATH "" FORCE)
elseif(HOLOVIZ_TEST_SLANG_MODE STREQUAL "external_system")
  set(_external_slang_root "${HOLOVIZ_TEST_CACHE_DIR}/external")
  write_test_slang_package("${_external_slang_root}" external_system)
  set(HOLOVIZ_SLANG_PROVIDER EXTERNAL)
  list(PREPEND CMAKE_PREFIX_PATH "${_external_slang_root}")
else()
  message(FATAL_ERROR "Unknown HOLOVIZ_TEST_SLANG_MODE: ${HOLOVIZ_TEST_SLANG_MODE}")
endif()

if(HOLOVIZ_TEST_MODE STREQUAL "embedded")
  set(holoscan_visualization_IS_TOP_LEVEL FALSE)
elseif(HOLOVIZ_TEST_MODE STREQUAL "holoscan_module")
  set(holoscan_visualization_IS_TOP_LEVEL TRUE)
  set(HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT holoscan-visualization-test)
  add_test_interface_target(holoviz_test_expected tl::expected)
  add_test_interface_target(holoviz_test_magic_enum magic_enum::magic_enum)
else()
  message(FATAL_ERROR "Unknown HOLOVIZ_TEST_MODE: ${HOLOVIZ_TEST_MODE}")
endif()

# Replace dependency acquisition with a local command stub while exercising the real recipes.
set(rapids-cmake-dir "${CMAKE_CURRENT_LIST_DIR}/stub_rapids")
list(APPEND CMAKE_MODULE_PATH "${HOLOSCAN_CMAKE_DIR}/modules")
include("${HOLOSCAN_CMAKE_DIR}/setup_dependencies.cmake")

include("${HOLOVIZ_CMAKE_DIR}/holoviz_dependencies.cmake")

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/holoviz_test_runtime.cpp"
  "void holoviz_test_runtime() {}\n")
add_library(holoviz_test_runtime SHARED
  "${CMAKE_CURRENT_BINARY_DIR}/holoviz_test_runtime.cpp")
holoviz_configure_runtime_search_path(holoviz_test_runtime)
get_target_property(_holoviz_test_install_rpath holoviz_test_runtime INSTALL_RPATH)
if(holoscan_visualization_IS_TOP_LEVEL AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(NOT _holoviz_test_install_rpath STREQUAL "\$ORIGIN")
    message(FATAL_ERROR
      "Top-level shared module install rpath is ${_holoviz_test_install_rpath}, expected \$ORIGIN")
  endif()
elseif(_holoviz_test_install_rpath)
  message(FATAL_ERROR
    "Dependency dispatch unexpectedly set install rpath ${_holoviz_test_install_rpath}")
endif()
unset(_holoviz_test_install_rpath)

get_property(_glfw_options GLOBAL PROPERTY HOLOVIZ_TEST_GLFW_OPTIONS)
foreach(_required_glfw_option IN ITEMS
    "GLFW_BUILD_EXAMPLES OFF"
    "GLFW_BUILD_TESTS OFF"
    "GLFW_BUILD_DOCS OFF"
    "GLFW_INSTALL OFF")
  if(NOT _required_glfw_option IN_LIST _glfw_options)
    message(FATAL_ERROR
      "GLFW dependency dispatch did not pass ${_required_glfw_option} to RAPIDS CPM")
  endif()
endforeach()
unset(_required_glfw_option)

if(holoscan_visualization_IS_TOP_LEVEL)
  if(NOT "GLFW_LIBRARY_TYPE STATIC" IN_LIST _glfw_options)
    message(FATAL_ERROR "Top-level dependency dispatch did not request a static CPM-built GLFW")
  endif()
elseif("GLFW_LIBRARY_TYPE STATIC" IN_LIST _glfw_options)
  message(FATAL_ERROR
    "Embedded dependency dispatch unexpectedly overrides the CPM-built GLFW library type")
endif()
unset(_glfw_options)

foreach(_required_target IN ITEMS
    CUDA::cuda_driver
    GPUOpen::VulkanMemoryAllocator
    Vulkan::Headers
    Vulkan::UtilityHeaders
    glfw
    magic_enum::magic_enum
    slang::slang
    stb::stb
    tl::expected
    volk::volk_headers)
  if(NOT TARGET ${_required_target})
    message(FATAL_ERROR "Dependency dispatch did not create ${_required_target}")
  endif()
endforeach()
unset(_required_target)

get_property(_imgui_source_dir GLOBAL PROPERTY HOLOVIZ_IMGUI_SOURCE_DIR)
if(NOT _imgui_source_dir)
  message(FATAL_ERROR "Dependency dispatch did not record the ImGui source directory")
endif()
unset(_imgui_source_dir)

if(HOLOVIZ_TEST_SLANG_MODE MATCHES "^external_")
  get_property(_slang_provider GLOBAL PROPERTY HOLOVIZ_TEST_SLANG_PROVIDER)
  if(NOT _slang_provider STREQUAL HOLOVIZ_TEST_SLANG_MODE)
    message(FATAL_ERROR
      "External Slang search resolved provider ${_slang_provider}, "
      "expected ${HOLOVIZ_TEST_SLANG_MODE}")
  endif()
  unset(_slang_provider)
else()
  get_property(_slang_force_download GLOBAL PROPERTY HOLOVIZ_TEST_SLANG_FORCE_DOWNLOAD)
  if(NOT _slang_force_download)
    message(FATAL_ERROR "Bundled Slang acquisition was not forced through RAPIDS CPM")
  endif()
  unset(_slang_force_download)
endif()

get_property(_rapids_calls GLOBAL PROPERTY HOLOVIZ_TEST_RAPIDS_CALLS)
set(_expected_rapids_calls glfw)
if(HOLOVIZ_TEST_SLANG_MODE STREQUAL "auto" OR
    HOLOVIZ_TEST_SLANG_MODE STREQUAL "prebuilt" OR
    HOLOVIZ_TEST_SLANG_MODE STREQUAL "source")
  list(APPEND _expected_rapids_calls slang)
endif()
list(APPEND _expected_rapids_calls
  VulkanHeaders
  VulkanUtilityLibraries
  volk
  VulkanMemoryAllocator
  imgui
  stb)
if(HOLOVIZ_TEST_MODE STREQUAL "embedded")
  list(PREPEND _expected_rapids_calls tl-expected magic_enum)
endif()
if(NOT _rapids_calls STREQUAL _expected_rapids_calls)
  message(FATAL_ERROR
    "Unexpected ${HOLOVIZ_TEST_MODE}/${HOLOVIZ_TEST_SLANG_MODE} dependency order: "
    "${_rapids_calls}; expected ${_expected_rapids_calls}")
endif()
