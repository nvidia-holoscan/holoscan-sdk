# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(holoviz
    holoviz.cpp
    ../../core/emitter_receiver_registry.cpp
)
target_link_libraries(holoviz_python
    PUBLIC
        holoscan::ops::holoviz
        magic_enum::magic_enum
)
