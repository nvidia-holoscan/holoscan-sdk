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

# Create examples
add_executable(multi_branch_pipeline
  multi_branch_pipeline.cpp
)
target_link_libraries(multi_branch_pipeline
  PRIVATE
  holoscan::core
)

# Copy config file to the build tree
add_custom_target(multi_branch_pipeline_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/multi_branch_pipeline.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "multi_branch_pipeline.yaml"
    BYPRODUCTS "multi_branch_pipeline.yaml"
)
add_dependencies(multi_branch_pipeline multi_branch_pipeline_yaml)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(multi_branch_pipeline PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the source
install(FILES multi_branch_pipeline.cpp
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the compiled example
install(TARGETS multi_branch_pipeline
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/multi_branch_pipeline.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_MULTI_BRANCH_OPERATOR_TEST
           COMMAND multi_branch_pipeline
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )

  # rx2 should receive close to 100 message (values 0 - 99)
  # rx1 will receive only some of these, but it is hard to know exactly which ones so just verify the first
  set_tests_properties(EXAMPLE_CPP_MULTI_BRANCH_OPERATOR_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "receiver 'rx2' received value: 90"
                       PASS_REGULAR_EXPRESSION "receiver 'rx1' received value: 0")
endif()
