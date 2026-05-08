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

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy Python application and YAML config to the build directory
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/pubsub_video_replayer.py"
         "${CMAKE_CURRENT_BINARY_DIR}/video_replayer.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/pubsub_video_replayer.py" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pubsub_video_replayer.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml"
  COMMENT "Copying pubsub_video_replayer.py and video_replayer.yaml"
)
add_custom_target(python_pubsub_video_replayer ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/pubsub_video_replayer.py"
          "${CMAKE_CURRENT_BINARY_DIR}/video_replayer.yaml"
)
add_dependencies(python_pubsub_video_replayer racerx_data)

# Install the app and config
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/pubsub_video_replayer.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)
