# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_executable(v4l2_camera
  v4l2_camera.cpp
)

target_link_libraries(v4l2_camera
  PRIVATE
  holoscan::core
  holoscan::ops::v4l2
  holoscan::ops::holoviz
)

# Copy config file
add_custom_target(v4l2_camera_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "v4l2_camera.yaml"
  BYPRODUCTS "v4l2_camera.yaml"
)

add_dependencies(v4l2_camera v4l2_camera_yaml)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(v4l2_camera PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the app
install(TARGETS v4l2_camera
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml"
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.cpp"
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
option(HOLOSCAN_BUILD_V4L2_TESTS "Build tests for V4L2 loopback" OFF)
if(HOLOSCAN_BUILD_TESTS AND HOLOSCAN_BUILD_V4L2_TESTS)
  # Assumes that the v4l2 video loopback is mounted on /dev/video3.  This allows us to create a
  # a virtual video device and stream data from an mp4 file without the need for a physical
  # video input device. To setup v4l2 video loopback, refer to the "Use with V4L2 Loopback Devices"
  # section of the README file for this example
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml CONFIG_STRING)
  string(REPLACE "device: \"/dev/video0\"" "device: \"/dev/video3\"" CONFIG_STRING "${CONFIG_STRING}")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_v4l2_camera_config.yaml)
  file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

  # Modify testcase to only run 10 frames
  add_custom_command(OUTPUT v4l2_camera_test.cpp
    PRE_LINK
    COMMAND sed 's/"visualizer",/"visualizer", make_condition<CountCondition>\(10\), /g'
            ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.cpp > v4l2_camera_test.cpp
  )

  # Create the test executable
  add_executable(v4l2_camera_test
    v4l2_camera_test.cpp
  )

  target_link_libraries(v4l2_camera_test
    PRIVATE
    holoscan::core
    holoscan::ops::v4l2
    holoscan::ops::holoviz
  )

  add_dependencies(v4l2_camera_test racerx_data)

  add_test(NAME EXAMPLE_CPP_V4L2_CAMERA_TEST
           COMMAND bash -c "ffmpeg -stream_loop -1 -re -i ${CMAKE_SOURCE_DIR}/data/racerx/racerx-small.mp4 \
                                    -pix_fmt yuyv422 -f v4l2 /dev/video3 &  sleep 5; \
                           ${CMAKE_CURRENT_BINARY_DIR}/v4l2_camera_test ${CONFIG_FILE}; echo 'Done'; kill %1"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_V4L2_CAMERA_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "Application has finished running")
endif()

