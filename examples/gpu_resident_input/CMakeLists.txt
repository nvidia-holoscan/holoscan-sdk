# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install following the relative folder path
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_CPP_EXAMPLES)
  # Create the GPU resident input executable
  add_executable(gpu_resident_input
    gpu_resident_input.cpp
    add_kernel.cu
  )
  target_link_libraries(gpu_resident_input
    PRIVATE
    holoscan::core
  )

  # Set the install RPATH based on the location of the Holoscan SDK libraries
  file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
  set_target_properties(gpu_resident_input PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

  if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
    # Install the source
    install(FILES
      gpu_resident_input.cpp
      add_kernel.cu.hpp
      add_kernel.cu
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
  install(TARGETS gpu_resident_input
    DESTINATION "${app_relative_dest_path}"
    COMPONENT holoscan-examples
  )

  # Testing
  if(HOLOSCAN_BUILD_TESTS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    add_test(NAME EXAMPLE_CPP_GPU_RESIDENT_INPUT_TEST
      COMMAND bash -c
        "${CMAKE_CURRENT_BINARY_DIR}/gpu_resident_input 2>&1 | \
        ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
          --pass-pattern 'verification complete' \
          --pass-pattern 'GPU Resident Performance Metrics' \
          --pass-pattern 'Average execution time:  [0-9]+\\.[0-9]+ us' \
          --pass-pattern 'Saved GPU-resident performance results as CSV to file' \
          --fail-pattern 'ERROR at index' \
          --fail-pattern 'Execution times are not available' \
          --fail-pattern 'Failed to open file'"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  endif()
endif()

install(
  FILES README.md
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)
