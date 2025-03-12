# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy native operator multi_branch_pipeline application
add_custom_target(python_multi_branch_pipeline ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multi_branch_pipeline.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "multi_branch_pipeline.py"
  BYPRODUCTS "multi_branch_pipeline.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/multi_branch_pipeline.py"
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
  add_test(NAME EXAMPLE_PYTHON_MULTI_BRANCH_TEST
    COMMAND python3 multi_branch_pipeline.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # rx2 should receive close to 100 message (values 0 - 99)
  # rx1 will receive only some of these, but it is hard to know exactly which ones so just verify the first
  set_tests_properties(EXAMPLE_PYTHON_MULTI_BRANCH_TEST PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Push failed on 'in'"
                       PASS_REGULAR_EXPRESSION "receiver 'rx2' received value: 90"
                       PASS_REGULAR_EXPRESSION "receiver 'rx1' received value: 0")

  add_test(NAME EXAMPLE_PYTHON_MULTI_BRANCH_EVENT_BASED_TEST
    COMMAND python3 multi_branch_pipeline.py --event_based
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  # rx2 should receive close to 100 message (values 0 - 99)
  # rx1 will receive only some of these, but it is hard to know exactly which ones so just verify the first
  set_tests_properties(EXAMPLE_PYTHON_MULTI_BRANCH_TEST PROPERTIES
                       FAIL_REGULAR_EXPRESSION "Push failed on 'in'"
                       PASS_REGULAR_EXPRESSION "receiver 'rx2' received value: 90"
                       PASS_REGULAR_EXPRESSION "receiver 'rx1' received value: 0")
endif()
