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

add_subdirectory(sequential)
add_subdirectory(conditional)
add_subdirectory(conditional_routing)
add_subdirectory(fork_join)
add_subdirectory(for_loop)
add_subdirectory(mixed_flows)
add_subdirectory(streaming)
add_subdirectory(streaming_with_monitor)
add_subdirectory(state_based_monitoring)

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

install(
  FILES README.md
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
