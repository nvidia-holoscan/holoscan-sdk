# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_custom_target(python_native_resource ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/native_resource.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "native_resource.py"
  BYPRODUCTS "native_resource.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/native_resource.py"
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
  add_test(NAME EXAMPLE_RESOURCES_PYTHON_NATIVE_RESOURCE_TEST
    COMMAND python3 native_resource.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_RESOURCES_PYTHON_NATIVE_RESOURCE_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "native resource setup method called"
    PASS_REGULAR_EXPRESSION "MinimalOp compute method called"
    FAIL_REGULAR_EXPRESSION "error"
    FAIL_REGULAR_EXPRESSION "Exception occurred"
  )
endif()
