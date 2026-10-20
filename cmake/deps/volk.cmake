# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(HOLOSCAN_VOLK_VERSION 1.4.350)  # tracks VK_HEADER_VERSION 350

if(TARGET volk::volk_headers)
  return()
endif()

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)
rapids_cpm_find(volk ${HOLOSCAN_VOLK_VERSION}
  GLOBAL_TARGETS volk::volk volk::volk_headers
  CPM_ARGS
    GITHUB_REPOSITORY zeux/volk
    GIT_TAG vulkan-sdk-${HOLOSCAN_VOLK_VERSION}.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL
  OPTIONS
    "VOLK_PULL_IN_VULKAN OFF"
    "VOLK_INSTALL OFF"
)
