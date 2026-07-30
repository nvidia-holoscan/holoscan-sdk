# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy Python application to the build directory
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/pubsub_ping_tensor.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pubsub_ping_tensor.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pubsub_ping_tensor.py"
  COMMENT "Copying pubsub_ping_tensor.py"
)
add_custom_target(python_pubsub_ping_tensor ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/pubsub_ping_tensor.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/pubsub_ping_tensor.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)
