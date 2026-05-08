# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_executable(pubsub_ping_tensor
  pubsub_ping_tensor.cpp
)
target_link_libraries(pubsub_ping_tensor
  PRIVATE
  holoscan::core
  holoscan::pubsub_runtime
  holoscan::pubsub_fastdds
  holoscan::ops::ping_tensor_tx
  holoscan::ops::ping_tensor_rx
)

file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(pubsub_ping_tensor PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
  install(
    FILES pubsub_ping_tensor.cpp
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
  )
endif()

install(TARGETS pubsub_ping_tensor
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
