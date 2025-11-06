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

# Copy native operator ping_multi_port application
add_custom_target(python_flow_tracker ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/flow_tracker.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "flow_tracker.py"
  BYPRODUCTS "flow_tracker.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/flow_tracker.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_FLOW_TRACKER_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/flow_tracker.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
        --pass-pattern 'Data Flow Tracking Results' \
        --pass-pattern 'Path 1: root1,middle1,leaf1' \
        --pass-pattern 'Path 2: root1,middle1,leaf2' \
        --pass-pattern 'Path 3: root2,middle2,leaf2'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
