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
add_custom_target(python_cuda_stream ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "cuda_stream.py"
  BYPRODUCTS "cuda_stream.py"
)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/cuda_stream.py"
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
  add_test(NAME EXAMPLE_PYTHON_CUDA_STREAM_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_stream.py --count=10 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received non-default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --pass-pattern 'Stream: 0x' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_DEFAULT_STREAM_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_stream.py --count=10 --default_stream 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --pass-pattern 'Stream: none' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_CUDA_STREAM_NO_ARRAY_INTERFACE_SYNC_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} cuda_stream.py --count=10 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'rx received non-default CUDA stream from port in' \
        --pass-pattern 'rx received message 10' \
        --pass-pattern 'Stream: 0x' \
        --fail-pattern 'rx received message 11'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_CUDA_STREAM_NO_ARRAY_INTERFACE_SYNC_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG;HOLOSCAN_CUDA_ARRAY_INTERFACE_SYNC=0;CUPY_CUDA_ARRAY_INTERFACE_SYNC=0"
  )

endif()
