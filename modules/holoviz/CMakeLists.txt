# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

project(holoviz
    DESCRIPTION "Holoviz"
    VERSION ${HOLOSCAN_BUILD_VERSION}
    LANGUAGES CXX CUDA
    )

add_subdirectory(thirdparty)

if(HOLOSCAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

add_subdirectory(src)
if(HOLOSCAN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
