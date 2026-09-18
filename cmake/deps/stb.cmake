# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# stb single-file image loaders for the HoloViz core examples. Header-only and source-only
# upstream (no CMake project or releases): download only and expose the source directory through
# one interface target. The revision is pinned to the SWIPAT-cleared commit; upgrading requires
# a new clearance.
set(HOLOSCAN_STB_GIT_TAG af1a5bc352164740c1cc1354942b1c6b72eacb8a)

if(TARGET stb::stb)
  return()
endif()

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)
set(_holoscan_stb_saved_download "${CPM_DOWNLOAD_stb}")
set(_holoscan_stb_download_defined FALSE)
if(DEFINED CPM_DOWNLOAD_stb)
  set(_holoscan_stb_download_defined TRUE)
endif()
set(CPM_DOWNLOAD_stb ON)
# stb has no versioned releases; the placeholder version is never matched against a package.
rapids_cpm_find(stb 0.0
  CPM_ARGS
    GITHUB_REPOSITORY nothings/stb
    GIT_TAG "${HOLOSCAN_STB_GIT_TAG}"
    DOWNLOAD_ONLY ON
)
if(_holoscan_stb_download_defined)
  set(CPM_DOWNLOAD_stb "${_holoscan_stb_saved_download}")
else()
  unset(CPM_DOWNLOAD_stb)
endif()
unset(_holoscan_stb_download_defined)
unset(_holoscan_stb_saved_download)

# Example-only dependency: the target is never installed or exported.
add_library(holoscan_stb INTERFACE)
target_include_directories(holoscan_stb INTERFACE
  $<BUILD_INTERFACE:${stb_SOURCE_DIR}>)
add_library(stb::stb ALIAS holoscan_stb)
