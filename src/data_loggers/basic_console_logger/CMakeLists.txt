# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_data_logger(basic_console_logger
    basic_console_logger.cpp
    gxf_console_logger.cpp
    simple_text_serializer.cpp
)

target_link_libraries(data_loggers_basic_console_logger
    PUBLIC
        holoscan::core
)
