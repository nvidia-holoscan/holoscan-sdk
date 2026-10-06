# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(cpp)
if(HOLOSCAN_BUILD_PYTHON_EXAMPLES)
    add_subdirectory(python)
endif()
