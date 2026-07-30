# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.rapids.ai/api/rapids-cmake/stable/command/rapids_cpm_find.html
include(${rapids-cmake-dir}/cpm/find.cmake)

set(ucxx_VERSION 0.48.00)
rapids_cpm_find(ucxx ${ucxx_VERSION}
    GLOBAL_TARGETS ucxx
    BUILD_EXPORT_SET ${HOLOSCAN_PACKAGE_NAME}-exports

    CPM_ARGS
        GITHUB_REPOSITORY rapidsai/ucxx
        GIT_TAG v${ucxx_VERSION}
        SOURCE_SUBDIR cpp

    OPTIONS
        "BUILD_TESTS OFF"
        "UCXX_ENABLE_RMM ON"
        "RMM_LOGGING_LEVEL INFO"
)
