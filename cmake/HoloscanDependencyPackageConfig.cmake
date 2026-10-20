# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(CMakePackageConfigHelpers)

function(holoscan_configure_dependency_package_config package_name install_destination output_variable)
  if(ARGC GREATER 3)
    set(HOLOSCAN_DEPENDENCY_INCLUDE_DIR "${ARGV3}")
  else()
    set(HOLOSCAN_DEPENDENCY_INCLUDE_DIR "${CMAKE_INSTALL_INCLUDEDIR}")
  endif()
  set(_output_dir "${CMAKE_CURRENT_BINARY_DIR}/install-package/dependencies/${package_name}")
  file(MAKE_DIRECTORY "${_output_dir}")
  set(_output "${_output_dir}/${package_name}-config.cmake")
  configure_package_config_file(
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/deps/configs/${package_name}-config.cmake.in"
    "${_output}"
    INSTALL_DESTINATION "${install_destination}"
    PATH_VARS CMAKE_INSTALL_INCLUDEDIR HOLOSCAN_DEPENDENCY_INCLUDE_DIR
  )
  set(${output_variable} "${_output}" PARENT_SCOPE)
endfunction()
