# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dependency resolution for SDK-embedded and standalone Holoscan Visualization builds.
#
# Dependency recipes stay in the central SDK catalog. Standalone Holoscan Visualization builds
# import shared targets from the installed SDK; SDK-embedded builds use the surrounding source
# tree's central recipes for their private dependencies.

include_guard(GLOBAL)

# On Windows there is no rpath: an executable loads dependency DLLs from its own directory, so
# tests and examples run from the build tree only if those DLLs are copied next to them. No-op
# elsewhere.
function(holoviz_copy_runtime_dlls target)
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:${target}> $<TARGET_FILE_DIR:${target}>
      COMMAND_EXPAND_LISTS
    )
  endif()
endfunction()

# A standalone Holoscan Visualization build may install shared dependencies beside libholoviz.
# Give this module-owned runtime the same sibling lookup used by the SDK build without changing
# the caller's directory-wide policy.
function(holoviz_configure_runtime_search_path target)
  get_target_property(_holoviz_target_type ${target} TYPE)
  if(holoscan_visualization_IS_TOP_LEVEL AND CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
      _holoviz_target_type STREQUAL "SHARED_LIBRARY")
    set_property(TARGET ${target} PROPERTY INSTALL_RPATH "\$ORIGIN")
  endif()
endfunction()

if(holoscan_visualization_IS_TOP_LEVEL)
  # The installed Holoscan package supplies the two shared targets used by this module.
  foreach(_holoviz_sdk_target IN ITEMS tl::expected magic_enum::magic_enum)
    if(NOT TARGET ${_holoviz_sdk_target})
      message(FATAL_ERROR
        "The installed Holoscan package did not provide ${_holoviz_sdk_target}")
    endif()
  endforeach()
  unset(_holoviz_sdk_target)
else()
  holoscan_add_dependency(expected_rapids)
  holoscan_add_dependency(magic_enum)
endif()

if(NOT TARGET CUDA::cuda_driver)
  find_package(CUDAToolkit REQUIRED)
endif()

set(_holoviz_dependencies
    glfw
    slang
    vulkan_headers
    vulkan_utility_libraries
    volk
    vma
    imgui)
if(HOLOSCAN_BUILD_EXAMPLES)
  # stb is only consumed by the core examples; skip acquisition otherwise.
  list(APPEND _holoviz_dependencies stb)
endif()
foreach(_holoviz_dependency IN LISTS _holoviz_dependencies)
  if(holoscan_visualization_IS_TOP_LEVEL)
    holoscan_add_dependency(${_holoviz_dependency}
      COMPONENT "${HOLOSCAN_VISUALIZATION_INSTALL_COMPONENT}")
  else()
    holoscan_add_dependency(${_holoviz_dependency})
  endif()
endforeach()
unset(_holoviz_dependency)
unset(_holoviz_dependencies)
