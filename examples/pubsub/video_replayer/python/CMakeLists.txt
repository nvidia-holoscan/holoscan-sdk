# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
