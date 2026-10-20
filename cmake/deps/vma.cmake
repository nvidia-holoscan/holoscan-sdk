# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_VMA_VERSION 3.4.0)

if(TARGET GPUOpen::VulkanMemoryAllocator)
  return()
endif()

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)
rapids_cpm_find(VulkanMemoryAllocator ${HOLOSCAN_VMA_VERSION}
  GLOBAL_TARGETS GPUOpen::VulkanMemoryAllocator
  CPM_ARGS
    GITHUB_REPOSITORY GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator
    GIT_TAG v${HOLOSCAN_VMA_VERSION}
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
)
