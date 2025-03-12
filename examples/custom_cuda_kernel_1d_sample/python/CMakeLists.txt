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

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(python_custom_cuda_kernel_1d_sample ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_1d_sample.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "custom_cuda_kernel_1d_sample.py"
  BYPRODUCTS "custom_cuda_kernel_1d_sample.py"
)

# Copy config file
add_custom_target(python_custom_cuda_kernel_1d_sample_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_1d_sample.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "custom_cuda_kernel_1d_sample.yaml"
  BYPRODUCTS "custom_cuda_kernel_1d_sample.yaml"
)

add_dependencies(python_custom_cuda_kernel_1d_sample python_custom_cuda_kernel_1d_sample_yaml)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_1d_sample.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_1d_sample.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_1d_sample.yaml CONFIG_STRING)
  string(REPLACE "count: 0" "count: 10" CONFIG_STRING "${CONFIG_STRING}")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/python_custom_cuda_kernel_1d_sample_config.yaml)
  file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

  # Add the test and make sure it runs
  add_test(NAME EXAMPLE_PYTHON_CUSTOM_CUDA_1D_SAMPLE_TEST
    COMMAND python3 custom_cuda_kernel_1d_sample.py --config python_custom_cuda_kernel_1d_sample_config.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_CUSTOM_CUDA_1D_SAMPLE_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )
endif()

