# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(ping_tx ping_tx.cpp)

target_link_libraries(op_ping_tx
    PUBLIC
        holoscan::core
)
