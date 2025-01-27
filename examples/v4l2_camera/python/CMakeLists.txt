# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  # Assumes that the v4l2 video loopback has already been mounted and the yaml files have been
  # updated to use the virtual loopback device.

  # Modify testcase to only run 10 frames
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/v4l2_camera.py PYTHON_SOURCE_STRING)
  string(REPLACE "from holoscan.core import Application"
      "from holoscan.conditions import CountCondition\nfrom holoscan.core import Application" PYTHON_SOURCE_STRING "${PYTHON_SOURCE_STRING}")
  string(REPLACE "self.add_flow" "visualizer.add_arg(CountCondition(self, 10)); self.add_flow" PYTHON_SOURCE_STRING "${PYTHON_SOURCE_STRING}")
  set(PYTHON_SOURCE_FILE ${CMAKE_CURRENT_BINARY_DIR}/v4l2_camera_test.py)
  file(WRITE ${PYTHON_SOURCE_FILE} "${PYTHON_SOURCE_STRING}")

  add_test(NAME EXAMPLE_PYTHON_V4L2_CAMERA_TEST
           COMMAND python3 v4l2_camera_test.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_V4L2_CAMERA_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Application has finished running"
  )

endif()
