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

# Copy native operator ping application
add_custom_target(python_ping_simple_thread_pool ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_simple_thread_pool.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "ping_simple_thread_pool.py"
  BYPRODUCTS "ping_simple_thread_pool.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ping_simple_thread_pool.py"
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
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME EXAMPLE_PYTHON_PING_SIMPLE_THREAD_POOL_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ping_simple_thread_pool.py 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
        --pass-pattern 'ThreadPool \\[cid: \\d+, name: pool1\\] created thread \\[uid: \\d+\\] for pinned entity \\[eid: \\d+, name: tx1\\]' \
        --pass-pattern 'ThreadPool \\[cid: \\d+, name: pool1\\] created thread \\[uid: \\d+\\] for pinned entity \\[eid: \\d+, name: rx1\\]' \
        --pass-pattern 'ThreadPool \\[cid: \\d+, name: pool2\\] created thread \\[uid: \\d+\\] for pinned entity \\[eid: \\d+, name: tx2\\]' \
        --pass-pattern 'ThreadPool \\[cid: \\d+, name: pool2\\] created thread \\[uid: \\d+\\] for pinned entity \\[eid: \\d+, name: rx2\\]' \
        --pass-pattern 'Rx message value: 10' \
        --pass-pattern 'Rx message value: 15' \
        --fail-pattern 'Rx message value: 16' \
        --fail-pattern 'error'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_SIMPLE_THREAD_POOL_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
  )
endif()
