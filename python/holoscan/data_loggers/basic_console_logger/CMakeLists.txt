# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

holoscan_pybind11_module(basic_console_logger
    basic_console_logger.cpp
)
target_link_libraries(basic_console_logger_python
    PUBLIC holoscan::data_loggers::basic_console_logger
)
