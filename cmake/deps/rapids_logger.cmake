# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Fine-grained control over the transitive source dependency "RAPIDS logger"
# needed by the RMM direct dependency. This module must be
# included before rmm.cmake so that RMM uses this pinned version.

set(rapids_logger_version 0.2.0)

include(FetchContent)

set(RAPIDS_LOGGER_GITHUB_REPOSITORY "https://github.com/rapidsai/rapids-logger.git")
set(RAPIDS_LOGGER_TAG "v${rapids_logger_version}")

FetchContent_Declare(
    rapids_logger
    GIT_REPOSITORY ${RAPIDS_LOGGER_GITHUB_REPOSITORY}
    GIT_TAG        ${RAPIDS_LOGGER_TAG}
    GIT_SHALLOW    TRUE
    PATCH_COMMAND patch -p1 -N -i ${CMAKE_CURRENT_LIST_DIR}/patches/rapids_logger.patch
    UPDATE_DISCONNECTED ON
)
FetchContent_MakeAvailable(rapids_logger)
