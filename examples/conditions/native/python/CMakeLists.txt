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

# Copy ping_periodic_native application
add_custom_target(python_ping_periodic_native ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_native.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "ping_periodic_native.py"
  BYPRODUCTS "ping_periodic_native.py"
)

# Install the ping_periodic_native app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_native.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Copy message_available_native application
add_custom_target(python_message_available_native ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/message_available_native.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "message_available_native.py"
  BYPRODUCTS "message_available_native.py"
)

# Install the message_available_native app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/message_available_native.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_PING_PERIODIC_NATIVE_TEST
    COMMAND python3 ping_periodic_native.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_PERIODIC_NATIVE_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "native condition setup method called"
    PASS_REGULAR_EXPRESSION "native condition initialize method called"
    PASS_REGULAR_EXPRESSION "native condition update_state method called"
    PASS_REGULAR_EXPRESSION "native condition check method called"
    PASS_REGULAR_EXPRESSION "native condition on_execute method called"
    PASS_REGULAR_EXPRESSION "Rx message value: 10"
  )

  add_test(NAME EXAMPLE_PYTHON_PING_MESSAGE_AVAILABLE_NATIVE_TEST
    COMMAND python3 message_available_native.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_MESSAGE_AVAILABLE_NATIVE_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
    PASS_REGULAR_EXPRESSION "Not adding default condition to port 'in' of operator 'rx'"
    PASS_REGULAR_EXPRESSION "native condition setup method called"
    PASS_REGULAR_EXPRESSION "native condition initialize method called"
    PASS_REGULAR_EXPRESSION "native condition update_state method called"
    PASS_REGULAR_EXPRESSION "native condition check method called"
    PASS_REGULAR_EXPRESSION "native condition on_execute method called"
    PASS_REGULAR_EXPRESSION "Rx message value: 10"
  )
endif()
