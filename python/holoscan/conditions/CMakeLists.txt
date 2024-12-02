# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

holoscan_pybind11_module(conditions
    asynchronous.cpp
    boolean.cpp
    conditions.cpp
    count.cpp
    cuda_buffer_available.cpp
    cuda_event.cpp
    cuda_stream.cpp
    downstream_message_affordable.cpp
    expiring_message.cpp
    message_available.cpp
    multi_message_available.cpp
    multi_message_available_timeout.cpp
    periodic.cpp
)
