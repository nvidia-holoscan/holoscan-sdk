# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Dear ImGui for the HoloViz text module. Source-only: the four core translation units are
# compiled directly into the HoloViz library, so no target is created or exported. The version is
# pinned to the SWIPAT-cleared release; upgrading requires a new clearance.
set(HOLOSCAN_IMGUI_VERSION 1.89.3)

get_property(_holoviz_imgui_source_dir GLOBAL PROPERTY HOLOVIZ_IMGUI_SOURCE_DIR)
if(_holoviz_imgui_source_dir)
  unset(_holoviz_imgui_source_dir)
  return()
endif()
unset(_holoviz_imgui_source_dir)

# Keep this recipe self-contained. RAPIDS guards this command module against repeated inclusion;
# the explicit include also lets configure-contract tests replace acquisition with a local stub.
include(${rapids-cmake-dir}/cpm/find.cmake)
set(_holoscan_imgui_saved_download "${CPM_DOWNLOAD_imgui}")
set(_holoscan_imgui_download_defined FALSE)
if(DEFINED CPM_DOWNLOAD_imgui)
  set(_holoscan_imgui_download_defined TRUE)
endif()
set(CPM_DOWNLOAD_imgui ON)
rapids_cpm_find(imgui "${HOLOSCAN_IMGUI_VERSION}"
  CPM_ARGS
    GITHUB_REPOSITORY ocornut/imgui
    GIT_TAG "v${HOLOSCAN_IMGUI_VERSION}"
    GIT_SHALLOW TRUE
    DOWNLOAD_ONLY ON
)
if(_holoscan_imgui_download_defined)
  set(CPM_DOWNLOAD_imgui "${_holoscan_imgui_saved_download}")
else()
  unset(CPM_DOWNLOAD_imgui)
endif()
unset(_holoscan_imgui_download_defined)
unset(_holoscan_imgui_saved_download)

# `holoscan_add_dependency()` is a function, so source-only CPM outputs otherwise disappear when
# the helper returns. A global property also provides one common contract for all build contexts.
set_property(GLOBAL PROPERTY HOLOVIZ_IMGUI_SOURCE_DIR "${imgui_SOURCE_DIR}")
