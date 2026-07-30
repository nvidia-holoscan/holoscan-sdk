# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Copy config file
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/app.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/app.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/app.yaml"
  COMMENT "Copying app.yaml"
)
add_custom_target(cli_packager_app_yaml ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/app.yaml"
)

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})
install(
  FILES README.md app.yaml
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
