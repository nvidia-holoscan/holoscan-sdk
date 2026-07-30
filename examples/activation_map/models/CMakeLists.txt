# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy dummy models
file(GLOB MODEL_ONNX_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/dummy_addition_model_*.onnx
)

set(MODEL_BASE_NAMES)
foreach(MODEL ${MODEL_ONNX_FILES})
  get_filename_component(BASE_NAME ${MODEL} NAME)
  list(APPEND MODEL_BASE_NAMES ${BASE_NAME})
endforeach()

add_custom_target(activation_map_dummy_models
  COMMAND ${CMAKE_COMMAND} -E copy ${MODEL_ONNX_FILES} ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${MODEL_BASE_NAMES}"
  BYPRODUCTS "${MODEL_BASE_NAMES}"
)

# Install dummy models
install(FILES
  ${MODEL_ONNX_FILES}
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
