# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy application file
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/v4l2_imx274_player.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.py"
  COMMENT "Copying v4l2_imx274_player.py"
)
add_custom_target(python_v4l2_imx274_player ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/v4l2_imx274_player.py"
)

# Copy config file
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/v4l2_imx274_player.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.yaml"
  COMMENT "Copying v4l2_imx274_player.yaml"
)
add_custom_target(python_v4l2_imx274_player_yaml
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/v4l2_imx274_player.yaml"
)

add_dependencies(python_v4l2_imx274_player
  python_v4l2_imx274_player_yaml
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.yaml"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
option(HOLOSCAN_BUILD_V4L2_IMX274_TESTS "Build tests for V4L2 IMX274" OFF)
if(HOLOSCAN_BUILD_TESTS AND HOLOSCAN_BUILD_V4L2_IMX274_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(NAME EXAMPLE_PYTHON_V4L2_IMX274_TEST
           COMMAND ${Python3_EXECUTABLE} v4l2_imx274_player.py --frame-limit 10
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_V4L2_IMX274_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Application has finished running"
  )

endif()
