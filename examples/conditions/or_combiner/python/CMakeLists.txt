# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy the sum-of-all application
add_custom_target(python_multi_port_or_combiner ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multi_port_or_combiner.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "multi_port_or_combiner.py"
  BYPRODUCTS "multi_port_or_combiner.py"
)

# Install the apps
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/multi_port_or_combiner.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_MULTI_PORT_OR_COMBINER_TEST
    COMMAND python3 multi_port_or_combiner.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_MULTI_PORT_OR_COMBINER_TEST PROPERTIES
    # count > 10 due to the OR condition
    PASS_REGULAR_EXPRESSION "count 15, message received"
    FAIL_REGULAR_EXPRESSION "count 21, message received"
  )
endif()
