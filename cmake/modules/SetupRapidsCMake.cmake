# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(rapids-cmake-version v26.02.00)

# https://github.com/rapidsai/rapids-cmake#installation
include(FetchContent)
FetchContent_Declare(rapids-cmake
  GIT_REPOSITORY https://github.com/rapidsai/rapids-cmake
  GIT_TAG ${rapids-cmake-version}
)
FetchContent_MakeAvailable(rapids-cmake)

include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-export)
include(rapids-find)
