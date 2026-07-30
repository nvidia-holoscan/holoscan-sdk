# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(ping_rx ping_rx.cpp)

target_link_libraries(op_ping_rx
    PUBLIC
        holoscan::core
)
