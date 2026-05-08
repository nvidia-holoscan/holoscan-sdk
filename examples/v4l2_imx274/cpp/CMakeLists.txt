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

# Create example
add_executable(v4l2_imx274_player
  v4l2_imx274_player.cpp
)

target_link_libraries(v4l2_imx274_player
  PRIVATE
  holoscan::core
  holoscan::ops::v4l2
  holoscan::ops::raw_image_processor
  holoscan::ops::bayer_demosaic
  holoscan::ops::holoviz
)

# Copy config file
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/v4l2_imx274_player.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.yaml"
  COMMENT "Copying v4l2_imx274_player.yaml"
)
add_custom_target(v4l2_imx274_player_yaml
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/v4l2_imx274_player.yaml"
)

add_dependencies(v4l2_imx274_player v4l2_imx274_player_yaml)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(v4l2_imx274_player PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the app
install(TARGETS v4l2_imx274_player
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.yaml"
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_imx274_player.cpp"
    DESTINATION ${app_relative_dest_path}
    COMPONENT holoscan-examples
  )

  # Install the minimal CMakeLists.txt file
  install(FILES CMakeLists.min.txt
    RENAME "CMakeLists.txt"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
  )
endif()

# Testing
option(HOLOSCAN_BUILD_V4L2_IMX274_TESTS "Build tests for V4L2 IMX274" OFF)
if(HOLOSCAN_BUILD_TESTS AND HOLOSCAN_BUILD_V4L2_IMX274_TESTS)
  add_test(NAME EXAMPLE_CPP_V4L2_IMX274_TEST
           COMMAND "${CMAKE_CURRENT_BINARY_DIR}/v4l2_imx274_player" --frame-limit 10
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_V4L2_IMX274_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Application has finished running")

endif()
