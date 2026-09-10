# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_subdirectory(unit)

if(NOT holoscan_visualization_IS_TOP_LEVEL)
    add_subdirectory(system)
endif()
