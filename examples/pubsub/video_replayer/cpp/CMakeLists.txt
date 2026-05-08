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

add_executable(pubsub_video_replayer
  pubsub_video_replayer.cpp
)

target_link_libraries(pubsub_video_replayer
  PRIVATE
  holoscan::core
  holoscan::pubsub_runtime
  holoscan::pubsub_fastdds
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
)

# Copy the YAML config next to the binary so the default config path resolution works.
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/video_replayer.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/video_replayer.yaml"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml"
  COMMENT "Copying video_replayer.yaml"
)
add_custom_target(pubsub_video_replayer_yaml
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/video_replayer.yaml"
)
add_dependencies(pubsub_video_replayer pubsub_video_replayer_yaml racerx_data)

file(RELATIVE_PATH install_lib_relative_path
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(pubsub_video_replayer
  PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
  install(
    FILES pubsub_video_replayer.cpp
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
  )
endif()

install(TARGETS pubsub_video_replayer
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
