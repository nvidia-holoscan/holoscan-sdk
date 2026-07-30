# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(gxf_codelet
    gxf_codelet.cpp
)
target_link_libraries(gxf_codelet_python
    PUBLIC holoscan::ops::gxf_codelet
)
