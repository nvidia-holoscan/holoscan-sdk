# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/activation_map.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.py"
  COMMENT "Copying activation_map.py"
)
add_custom_target(python_activation_map ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/activation_map.py"
)

# Copy config file
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/activation_map.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.yaml"
  COMMENT "Copying activation_map.yaml"
)
add_custom_target(python_activation_map_yaml
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/activation_map.yaml"
)

add_dependencies(python_activation_map python_activation_map_yaml)

# Install the configuration file
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/activation_map.yaml"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT "holoscan-examples"
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(NAME EXAMPLE_PYTHON_ACTIVATION_MAP_TEST
    COMMAND bash -c
      "${Python3_EXECUTABLE} activation_map.py --config ${CMAKE_CURRENT_BINARY_DIR}/activation_map.yaml 2>&1 | \
       ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py \
         --pass-pattern 'Model first, second, third was selected to infer' \
         --pass-pattern '7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7' \
         --pass-pattern '8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8' \
         --pass-pattern '9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )
endif()
