# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(async_ping_rx async_ping_rx.cpp)

target_link_libraries(op_async_ping_rx
    PUBLIC
        holoscan::core
)
