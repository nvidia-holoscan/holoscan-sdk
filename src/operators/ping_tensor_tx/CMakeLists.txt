# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_operator(ping_tensor_tx ping_tensor_tx.cpp)

target_link_libraries(op_ping_tensor_tx
    PUBLIC
        holoscan::core
)
