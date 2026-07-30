# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy ping_periodic_native application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ping_periodic_native.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_native.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_native.py"
  COMMENT "Copying ping_periodic_native.py"
)
add_custom_target(python_ping_periodic_native ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ping_periodic_native.py"
)

# Install the ping_periodic_native app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/ping_periodic_native.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Copy message_available_native application
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/message_available_native.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/message_available_native.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/message_available_native.py"
  COMMENT "Copying message_available_native.py"
)
add_custom_target(python_message_available_native ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/message_available_native.py"
)

# Install the message_available_native app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/message_available_native.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME EXAMPLE_PYTHON_PING_PERIODIC_NATIVE_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} ping_periodic_native.py 2>&1 | \
       ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
         --pass-pattern 'native condition setup method called' \
         --pass-pattern 'native condition initialize method called' \
         --pass-pattern 'native condition update_state method called' \
         --pass-pattern 'native condition check method called' \
         --pass-pattern 'native condition on_execute method called' \
         --pass-pattern 'Rx message value: 10'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_test(NAME EXAMPLE_PYTHON_PING_MESSAGE_AVAILABLE_NATIVE_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} message_available_native.py 2>&1 | \
       ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
         --pass-pattern 'Not adding default condition to port in of operator rx' \
         --pass-pattern 'native condition setup method called' \
         --pass-pattern 'native condition initialize method called' \
         --pass-pattern 'native condition update_state method called' \
         --pass-pattern 'native condition check method called' \
         --pass-pattern 'native condition on_execute method called' \
         --pass-pattern 'Rx message value: 10'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_PING_MESSAGE_AVAILABLE_NATIVE_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_LOG_LEVEL=DEBUG"
  )
endif()
