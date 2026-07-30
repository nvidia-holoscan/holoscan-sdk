# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy configuration file into the build tree
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/identity_model.onnx"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/identity_model.onnx" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/identity_model.onnx"
  COMMENT "Copying identity_model.onnx"
)
add_custom_target(identity_model_onnx ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/identity_model.onnx"
)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/identity_model.onnx"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
