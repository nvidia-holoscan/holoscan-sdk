# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_holoscan_data_logger(async_console_logger
    async_console_backend.cpp
    async_console_logger.cpp
)

target_link_libraries(data_loggers_async_console_logger
    PUBLIC
        holoscan::core
        holoscan::data_loggers::basic_console_logger
)

# Add concurrent_queue header-only library include directory
target_include_directories(data_loggers_async_console_logger
    PRIVATE
        $<BUILD_INTERFACE:${concurrent_queue_SOURCE_DIR}>
)
