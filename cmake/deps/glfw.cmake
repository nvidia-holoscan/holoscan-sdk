# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_GLFW_VERSION 3.4)

if(TARGET glfw)
  return()
endif()

# GLFW is absorbed into libholoviz as a static archive in every configuration. holoviz links it
# PRIVATE; a static dependency is neither propagated to consumers nor recorded in the exported
# target's IMPORTED_LINK_DEPENDENT_LIBRARIES, so glfw needs no export set and never becomes a
# runtime dependency of the SDK or module package. holoviz links with --exclude-libs,ALL, so the
# archive's symbols stay hidden. An existing parent-provided target remains untouched.
set(_holoscan_glfw_options
  "GLFW_BUILD_EXAMPLES OFF"
  "GLFW_BUILD_TESTS OFF"
  "GLFW_BUILD_DOCS OFF"
  "GLFW_INSTALL OFF"
  "GLFW_LIBRARY_TYPE STATIC"
)

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
