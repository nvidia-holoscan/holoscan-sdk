# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cuda_green_context.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cuda_green_context.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_green_context.py"
  COMMENT "Copying cuda_green_context.py"
)
add_custom_target(python_cuda_green_context ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/cuda_green_context.py"
)
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/cuda_default_green_context.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cuda_default_green_context.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cuda_default_green_context.py"
  COMMENT "Copying cuda_default_green_context.py"
)
add_custom_target(python_default_cuda_green_context ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/cuda_default_green_context.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_green_context.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_default_green_context.py"
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
  add_test(NAME EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_DEFAULT_STREAM_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_green_context.py --count=10 -d 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_DEFAULT_STREAM_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_NON_DEFAULT_STREAM_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_green_context.py --count=10 -g 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received non-default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CUDA_GREEN_CONTEXT_NON_DEFAULT_STREAM_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_DEFAULT_GREEN_CONTEXT_POOL_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_default_green_context.py --count=10 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received non-default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CUDA_DEFAULT_GREEN_CONTEXT_POOL_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_DEFAULT_GREEN_CONTEXT_POOL_DEFAULT_STREAM_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_default_green_context.py --count=10 -d 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CUDA_DEFAULT_GREEN_CONTEXT_POOL_DEFAULT_STREAM_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
  )
endif()
