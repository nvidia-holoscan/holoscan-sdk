# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(bayer_demosaic
    bayer_demosaic.cpp
)
target_link_libraries(bayer_demosaic_python
    PUBLIC holoscan::ops::bayer_demosaic
)
