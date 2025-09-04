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
add_custom_target(python_cuda_green_context ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cuda_green_context.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "cuda_green_context.py"
  BYPRODUCTS "cuda_green_context.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_green_context.py"
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
  add_test(NAME EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_TEST
    COMMAND python3 cuda_green_context.py --count=10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
    PASS_REGULAR_EXPRESSION "rx received non-default CUDA stream from port 'in'"
    PASS_REGULAR_EXPRESSION "rx received message 10"
    FAIL_REGULAR_EXPRESSION "rx received message 11"
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_NO_ARRAY_INTERFACE_SYNC_TEST
    COMMAND python3 cuda_green_context.py --count=10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_NO_ARRAY_INTERFACE_SYNC_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG;HOLOSCAN_CUDA_ARRAY_INTERFACE_SYNC=0;CUPY_CUDA_ARRAY_INTERFACE_SYNC=0"
    PASS_REGULAR_EXPRESSION "rx received non-default CUDA stream from port 'in'"
    PASS_REGULAR_EXPRESSION "rx received message 10"
    FAIL_REGULAR_EXPRESSION "rx received message 11"
  )

endif()
