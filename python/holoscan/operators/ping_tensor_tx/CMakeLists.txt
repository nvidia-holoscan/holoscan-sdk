# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(ping_tensor_tx
    ping_tensor_tx.cpp
)
target_link_libraries(ping_tensor_tx_python
    PUBLIC holoscan::ops::ping_tensor_tx
)
