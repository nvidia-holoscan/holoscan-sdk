# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create examples
add_executable(benchmark_message_passing
  benchmark_message_passing.cpp
)

target_link_libraries(benchmark_message_passing
  PRIVATE
  holoscan::core
)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(benchmark_message_passing PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES benchmark_message_passing.cpp
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

# Install the compiled example
install(TARGETS benchmark_message_passing
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_BENCHMARK_MESSAGE_PASSING_TEST
           COMMAND benchmark_message_passing
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_MESSAGE_PASSING_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "\\| Trial \\|"
                       TIMEOUT 120
                      )

  add_test(NAME EXAMPLE_CPP_BENCHMARK_MESSAGE_PASSING_COMPONENTS_TEST
           COMMAND benchmark_message_passing
                   --add-timestamp
                   --add-cuda-stream
                   --add-metadata
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_MESSAGE_PASSING_COMPONENTS_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "\\| Trial \\|"
                       TIMEOUT 120
                      )
endif()
