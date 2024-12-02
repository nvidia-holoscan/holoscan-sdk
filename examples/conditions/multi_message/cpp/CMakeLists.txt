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
add_executable(multi_message_per_receiver
  multi_message_per_receiver.cpp
  common_ops.hpp
)
target_link_libraries(multi_message_per_receiver
  PRIVATE
  holoscan::core
)

add_executable(multi_message_sum_of_all
  multi_message_sum_of_all.cpp
  common_ops.hpp
)
target_link_libraries(multi_message_sum_of_all
  PRIVATE
  holoscan::core
)

add_executable(single_message_timeout
  single_message_timeout.cpp
  common_ops.hpp
)
target_link_libraries(single_message_timeout
  PRIVATE
  holoscan::core
)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(multi_message_per_receiver PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")
set_target_properties(multi_message_sum_of_all PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")
set_target_properties(single_message_timeout PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES multi_message_per_receiver.cpp multi_message_sum_of_all.cpp single_message_timeout.cpp common_ops.hpp
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
endif()

# Install the compiled examples
install(TARGETS multi_message_per_receiver multi_message_sum_of_all single_message_timeout
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_MULTI_MESSAGE_PER_RECEIVER_TEST
           COMMAND multi_message_per_receiver
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_MULTI_MESSAGE_PER_RECEIVER_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "message received on in1: Hello from tx1"
                       PASS_REGULAR_EXPRESSION "message received on in2: Hello from tx2"
                       PASS_REGULAR_EXPRESSION "message received on in3: Hello from tx3"
                       FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
                      )

  add_test(NAME EXAMPLE_CPP_MULTI_MESSAGE_SUM_OF_ALL_TEST
           COMMAND multi_message_sum_of_all
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(
    EXAMPLE_CPP_MULTI_MESSAGE_SUM_OF_ALL_TEST PROPERTIES
    # check that multiple messages arrived on all ports
    # (expect several message on in1, 4-5 on in2 and at least one on in3 with the timings specified)
    PASS_REGULAR_EXPRESSION "messages received on in1: \\[\"tx1\", \"tx1\", \"tx1\", \"tx1\""
    PASS_REGULAR_EXPRESSION "messages received on in2: \\[\"tx2\", \"tx2\", \"tx2\""
    PASS_REGULAR_EXPRESSION "messages received on in3: \\[\"tx3\""
    # with given periodic conditions should not be possible that tx3 arrived >3 this times
    FAIL_REGULAR_EXPRESSION "messages received on in3: \\[\"tx3\", \"tx3\", \"tx3\", \"tx3\""
    FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
  )

  add_test(NAME EXAMPLE_CPP_SINGLE_MESSAGE_TIMEOUT_TEST
           COMMAND single_message_timeout
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(
    EXAMPLE_CPP_SINGLE_MESSAGE_TIMEOUT_TEST PROPERTIES
    # check that despite min_sum=5, only 3 messages arrived due to the execution_frequency
    PASS_REGULAR_EXPRESSION "3 messages received on in"
    FAIL_REGULAR_EXPRESSION "Unable to convert argument type"
  )
endif()
