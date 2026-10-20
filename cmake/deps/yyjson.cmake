# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(${rapids-cmake-dir}/cpm/find.cmake)

# yyjson is a source-only, private config_io parser backend. Force the pinned upstream source
# instead of accepting a system package so Holoscan controls the implementation compiled into the
# component without exporting a yyjson target, header, archive, or downstream package dependency.
set(yyjson_VERSION 0.12.0)
set(_holoscan_yyjson_saved_download "${CPM_DOWNLOAD_yyjson}")
set(_holoscan_yyjson_had_download FALSE)
if(DEFINED CPM_DOWNLOAD_yyjson)
  set(_holoscan_yyjson_had_download TRUE)
endif()
set(CPM_DOWNLOAD_yyjson ON)
rapids_cpm_find(yyjson "${yyjson_VERSION}"
    CPM_ARGS
        GITHUB_REPOSITORY ibireme/yyjson
        GIT_TAG "${yyjson_VERSION}"
        GIT_SHALLOW TRUE
        DOWNLOAD_ONLY ON
)
# `holoscan_add_dependency()` is a function, so source-only CPM outputs otherwise disappear when the
# helper returns. Propagate only the private source location needed by the owning CMake scope.
set(yyjson_SOURCE_DIR "${yyjson_SOURCE_DIR}" PARENT_SCOPE)
if(_holoscan_yyjson_had_download)
  set(CPM_DOWNLOAD_yyjson "${_holoscan_yyjson_saved_download}")
else()
  unset(CPM_DOWNLOAD_yyjson)
endif()
unset(_holoscan_yyjson_had_download)
unset(_holoscan_yyjson_saved_download)
