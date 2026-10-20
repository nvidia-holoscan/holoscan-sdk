# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Pinned to the same Vulkan SDK release volk tracks (see volk.cmake). HoloViz never links the
# Vulkan loader, so only these headers are needed. Pinning keeps the build independent of older
# system Vulkan headers.
set(HOLOSCAN_VULKAN_HEADERS_VERSION 1.4.350)

if(TARGET Vulkan::Headers)
  return()
endif()

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)
rapids_cpm_find(VulkanHeaders ${HOLOSCAN_VULKAN_HEADERS_VERSION}
  GLOBAL_TARGETS Vulkan::Headers
  CPM_ARGS
    GITHUB_REPOSITORY KhronosGroup/Vulkan-Headers
    GIT_TAG vulkan-sdk-${HOLOSCAN_VULKAN_HEADERS_VERSION}.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)
