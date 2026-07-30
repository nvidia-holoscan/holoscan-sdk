# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Create examples
add_executable(ping_distributed
  ping_distributed.cpp
)
target_link_libraries(ping_distributed
  PUBLIC
  holoscan::core
  holoscan::ops::ping_tensor_tx
  holoscan::ops::ping_tensor_rx
)

# Install examples

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(ping_distributed PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(
  FILES
    ping_distributed.cpp
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
install(TARGETS ping_distributed
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  # For iGPU, a call to `ucp_init_version` with CUDA_VISIBLE_DEVICES="" seems to cause a segfault.
  # Limit the following test case to x86_64 systems to avoid this.
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    # emulate a x86_64 system without any GPUs by setting CUDA_VISIBLE_DEVICES=""
    add_test(NAME EXAMPLE_CPP_PING_DISTRIBUTED_CPU_ONLY_TEST
      COMMAND bash -c
        "${CMAKE_CURRENT_BINARY_DIR}/ping_distributed 2>&1 | \
        ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
          --pass-pattern 'no CUDA-capable device is detected' \
          --pass-pattern 'rx received message 10: Tensor key: tensor' \
          --fail-pattern 'initialized independent of a parent entity'"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )
    set_tests_properties(EXAMPLE_CPP_PING_DISTRIBUTED_CPU_ONLY_TEST PROPERTIES
      RESOURCE_LOCK "distributed_ucx"
      ENVIRONMENT "CUDA_VISIBLE_DEVICES="
    )
  endif()

  # test with CPU tensors, but don't explicitly make the GPU not visible
  add_test(NAME EXAMPLE_CPP_PING_DISTRIBUTED_CPU_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_PING_DISTRIBUTED_CPU_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "rx received message 10: Tensor key: 'tensor', shape: \\(32, 64\\)"
    RESOURCE_LOCK "distributed_ucx"
  )

  # test with GPU tensors
  add_test(NAME EXAMPLE_CPP_PING_DISTRIBUTED_GPU_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed --gpu
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_PING_DISTRIBUTED_GPU_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "rx received message 10: Tensor key: 'tensor', shape: \\(32, 64\\)"
    FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
    RESOURCE_LOCK "distributed_ucx"
  )

  add_test(NAME EXAMPLE_CPP_PING_DISTRIBUTED_FLOW_TRACKING_TEST
    COMMAND bash -c
      "${CMAKE_CURRENT_BINARY_DIR}/ping_distributed --track 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'Data Flow Tracking Results' \
        --pass-pattern 'Fragment: fragment1' \
        --pass-pattern 'Fragment: fragment2'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_PING_DISTRIBUTED_FLOW_TRACKING_TEST PROPERTIES
    RESOURCE_LOCK "distributed_ucx"
  )

  add_test(NAME EXAMPLE_CPP_PING_DISTRIBUTED_HELP_STRING_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/ping_distributed --help
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_PING_DISTRIBUTED_HELP_STRING_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "ping_distributed \\[OPTIONS\\]"
    FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
  )
endif()
