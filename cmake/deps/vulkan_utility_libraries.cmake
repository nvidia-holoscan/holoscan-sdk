# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Header-only utilities. Keep this at the same Vulkan SDK release as Vulkan::Headers, which must
# be added before this recipe.
set(HOLOSCAN_VULKAN_UTILITY_LIBRARIES_VERSION 1.4.350)

if(TARGET Vulkan::UtilityHeaders)
  return()
endif()

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)
rapids_cpm_find(VulkanUtilityLibraries ${HOLOSCAN_VULKAN_UTILITY_LIBRARIES_VERSION}
  GLOBAL_TARGETS Vulkan::UtilityHeaders
  CPM_ARGS
    GITHUB_REPOSITORY KhronosGroup/Vulkan-Utility-Libraries
    GIT_TAG vulkan-sdk-${HOLOSCAN_VULKAN_UTILITY_LIBRARIES_VERSION}.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)
