# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create examples
add_executable(benchmark_scheduler_throughput
  benchmark_scheduler_throughput.cpp
)

target_link_libraries(benchmark_scheduler_throughput
  PRIVATE
  holoscan::core
)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR}
                   ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(benchmark_scheduler_throughput PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES benchmark_scheduler_throughput.cpp
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
install(TARGETS benchmark_scheduler_throughput
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_TEST
           COMMAND benchmark_scheduler_throughput
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "\\| Trial \\|"
                       TIMEOUT 120
                      )

  add_test(NAME EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_FLAGS_TEST
           COMMAND benchmark_scheduler_throughput
                   --enable_queue_stealing
                   --enable_postcheck_fastpath
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
          )
  set_tests_properties(EXAMPLE_CPP_BENCHMARK_SCHEDULER_THROUGHPUT_FLAGS_TEST PROPERTIES
                       PASS_REGULAR_EXPRESSION "\\| Trial \\|"
                       TIMEOUT 120
                      )
endif()
