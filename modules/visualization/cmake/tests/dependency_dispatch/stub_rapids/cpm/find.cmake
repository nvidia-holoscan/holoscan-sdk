# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(rapids_cpm_find package_name package_version)
  cmake_parse_arguments(PARSE_ARGV 2 _rapids "" "" "GLOBAL_TARGETS;CPM_ARGS;OPTIONS")

  get_property(_rapids_calls GLOBAL PROPERTY HOLOVIZ_TEST_RAPIDS_CALLS)
  list(APPEND _rapids_calls "${package_name}")
  set_property(GLOBAL PROPERTY HOLOVIZ_TEST_RAPIDS_CALLS "${_rapids_calls}")

  if(package_name STREQUAL "tl-expected")
    add_library(holoviz_test_expected INTERFACE)
    add_library(tl::expected ALIAS holoviz_test_expected)
    set(tl-expected_ADDED FALSE PARENT_SCOPE)
  elseif(package_name STREQUAL "magic_enum")
    add_library(holoviz_test_magic_enum INTERFACE)
    add_library(magic_enum::magic_enum ALIAS holoviz_test_magic_enum)
    set(magic_enum_ADDED FALSE PARENT_SCOPE)
  elseif(package_name STREQUAL "glfw")
    set_property(GLOBAL PROPERTY HOLOVIZ_TEST_GLFW_OPTIONS "${_rapids_OPTIONS}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/holoviz_test_glfw.cpp" "void glfw_stub() {}\n")
    add_library(glfw SHARED "${CMAKE_CURRENT_BINARY_DIR}/holoviz_test_glfw.cpp")
  elseif(package_name STREQUAL "slang")
    set_property(GLOBAL PROPERTY HOLOVIZ_TEST_SLANG_FORCE_DOWNLOAD "${CPM_DOWNLOAD_slang}")
    set(_slang_source_dir "${CMAKE_CURRENT_BINARY_DIR}/holoviz_test_slang_package")
    file(MAKE_DIRECTORY "${_slang_source_dir}/lib/cmake/slang")
    file(WRITE "${_slang_source_dir}/lib/cmake/slang/slangConfig.cmake" "")
    file(WRITE "${_slang_source_dir}/lib/cmake/slang/slangConfigVersion.cmake"
      "set(PACKAGE_VERSION 2026.14)\n"
      "set(PACKAGE_VERSION_COMPATIBLE TRUE)\n"
      "if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)\n"
      "  set(PACKAGE_VERSION_EXACT TRUE)\n"
      "elseif(PACKAGE_FIND_VERSION VERSION_GREATER PACKAGE_VERSION)\n"
      "  set(PACKAGE_VERSION_COMPATIBLE FALSE)\n"
      "endif()\n")
    set(slang_SOURCE_DIR "${_slang_source_dir}" PARENT_SCOPE)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/holoviz_test_slang.cpp" "void slang_stub() {}\n")
    add_library(slang SHARED "${CMAKE_CURRENT_BINARY_DIR}/holoviz_test_slang.cpp")
    add_library(slang::slang ALIAS slang)
  elseif(package_name STREQUAL "VulkanHeaders")
    add_library(holoviz_test_vulkan_headers INTERFACE)
    add_library(Vulkan::Headers ALIAS holoviz_test_vulkan_headers)
  elseif(package_name STREQUAL "VulkanUtilityLibraries")
    add_library(holoviz_test_vulkan_utilities INTERFACE)
    add_library(Vulkan::UtilityHeaders ALIAS holoviz_test_vulkan_utilities)
  elseif(package_name STREQUAL "volk")
    add_library(holoviz_test_volk INTERFACE)
    add_library(volk::volk ALIAS holoviz_test_volk)
    add_library(volk::volk_headers ALIAS holoviz_test_volk)
  elseif(package_name STREQUAL "VulkanMemoryAllocator")
    add_library(holoviz_test_vma INTERFACE)
    add_library(GPUOpen::VulkanMemoryAllocator ALIAS holoviz_test_vma)
  elseif(package_name STREQUAL "imgui")
    set(imgui_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/stub_imgui" PARENT_SCOPE)
  elseif(package_name STREQUAL "stb")
    set(stb_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/stub_stb" PARENT_SCOPE)
  else()
    message(FATAL_ERROR
      "Unexpected rapids_cpm_find package ${package_name} ${package_version}")
  endif()
endfunction()
