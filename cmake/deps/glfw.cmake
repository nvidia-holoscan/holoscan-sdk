# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_GLFW_VERSION 3.4)

if(TARGET glfw)
  return()
endif()

set(_holoscan_glfw_options
  "GLFW_BUILD_EXAMPLES OFF"
  "GLFW_BUILD_TESTS OFF"
  "GLFW_BUILD_DOCS OFF"
  "GLFW_INSTALL OFF"
)

# Preserve the fetched-dependency policy of top-level Holoscan Visualization builds: GLFW is absorbed into the
# backend rather than becoming another runtime dependency of the module package. An existing
# parent-provided target remains untouched.
if(holoscan_visualization_IS_TOP_LEVEL)
  list(APPEND _holoscan_glfw_options "GLFW_LIBRARY_TYPE STATIC")
endif()

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)
rapids_cpm_find(glfw ${HOLOSCAN_GLFW_VERSION}
  GLOBAL_TARGETS glfw
  CPM_ARGS
    GITHUB_REPOSITORY glfw/glfw
    GIT_TAG ${HOLOSCAN_GLFW_VERSION}
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
  OPTIONS
    ${_holoscan_glfw_options}
)
unset(_holoscan_glfw_options)

if(NOT holoscan_visualization_IS_TOP_LEVEL)
  install(TARGETS glfw
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    COMPONENT ${HOLOSCAN_DEPEND_INSTALL_COMPONENT}
  )
endif()
