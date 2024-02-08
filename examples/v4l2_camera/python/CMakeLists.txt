# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy v4l2_camera application file
add_custom_target(python_v4l2_camera ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "v4l2_camera.py"
  BYPRODUCTS "v4l2_camera.py"
)

# Copy config file
add_custom_target(python_v4l2_camera_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "v4l2_camera.yaml"
  BYPRODUCTS "v4l2_camera.yaml"
)

add_dependencies(python_v4l2_camera python_v4l2_camera_yaml)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
option(HOLOSCAN_BUILD_V4L2_TESTS "Build tests for V4L2 loopback" OFF)
if(HOLOSCAN_BUILD_TESTS AND HOLOSCAN_BUILD_V4L2_TESTS)
  # Assumes that the v4l2 video loopback is mounted on /dev/video3.  This allows us to create a
  # a virtual video device and stream data from an mp4 file without the need for a physical
  # video input device. To setup v4l2 video loopback, refer to the "Use with V4L2 Loopback Devices"
  # section of the README file for this example
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.yaml CONFIG_STRING)
  string(REPLACE "device: \"/dev/video0\"" "device: \"/dev/video3\"" CONFIG_STRING "${CONFIG_STRING}")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/python_v4l2_camera_config.yaml)
  file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

  # Modify testcase to only run 10 frames
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.py PYTHON_SOURCE_STRING)
  string(REPLACE "from holoscan.core import Application"
      "from holoscan.conditions import CountCondition\nfrom holoscan.core import Application" PYTHON_SOURCE_STRING "${PYTHON_SOURCE_STRING}")
  string(REPLACE "self.add_flow" "visualizer.add_arg(CountCondition(self, 10)); self.add_flow" PYTHON_SOURCE_STRING "${PYTHON_SOURCE_STRING}")
  set(PYTHON_SOURCE_FILE ${CMAKE_CURRENT_BINARY_DIR}/v4l2_camera_test.py)
  file(WRITE ${PYTHON_SOURCE_FILE} "${PYTHON_SOURCE_STRING}")

  add_test(NAME EXAMPLE_PYTHON_V4L2_CAMERA_TEST
           COMMAND bash -c "ffmpeg -stream_loop -1 -re -i ${CMAKE_SOURCE_DIR}/data/racerx/racerx-small.mp4 \
                                    -pix_fmt yuyv422 -f v4l2 /dev/video3 &  sleep 5; \
                           python3 v4l2_camera_test.py --config ${CONFIG_FILE}; echo 'Done'; kill %1"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_V4L2_CAMERA_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Application has finished running"
  )
endif()
