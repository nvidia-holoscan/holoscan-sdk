# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holoscan_data_logger(async_console_logger
    async_console_backend.cpp
    async_console_logger.cpp
)

target_link_libraries(data_loggers_async_console_logger
    PUBLIC
        holoscan::core
        holoscan::data_loggers::basic_console_logger
)
