# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_custom_target(python_ping_simple ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_simple.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "ping_simple.py"
  BYPRODUCTS "ping_simple.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ping_simple.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_PING_SIMPLE_TEST
    COMMAND python3 ping_simple.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_SIMPLE_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Rx message value: 10"
  )
endif()
