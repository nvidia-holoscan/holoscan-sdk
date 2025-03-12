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

add_executable(custom_cuda_kernel_multi_sample
    main.cpp
)

target_link_libraries(custom_cuda_kernel_multi_sample
    PRIVATE
    holoscan::core
    holoscan::ops::video_stream_replayer
    holoscan::ops::format_converter
    holoscan::ops::inference_processor
    holoscan::ops::holoviz
)

# Copy config file
add_custom_target(custom_cuda_kernel_multi_sample_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_multi_sample.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "custom_cuda_kernel_multi_sample.yaml"
    BYPRODUCTS "custom_cuda_kernel_multi_sample.yaml"
)
add_dependencies(custom_cuda_kernel_multi_sample custom_cuda_kernel_multi_sample_yaml racerx_data)

file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(custom_cuda_kernel_multi_sample PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the app
install(TARGETS "custom_cuda_kernel_multi_sample"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
)
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_multi_sample.yaml"
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    DESTINATION ${app_relative_dest_path}
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
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/custom_cuda_kernel_multi_sample.yaml CONFIG_STRING)
  string(REPLACE "count: 0" "count: 10" CONFIG_STRING "${CONFIG_STRING}")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_custom_cuda_kernel_multi_sample_config.yaml)
  file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

  # Add the test and make sure it runs
  add_test(NAME EXAMPLE_CPP_CUSTOM_CUDA_MULTI_SAMPLE_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/custom_cuda_kernel_multi_sample ${CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_CUSTOM_CUDA_MULTI_SAMPLE_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )
endif()
