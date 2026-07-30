# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Copy distributed_pose_tree application file
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/distributed_pose_tree.py"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/distributed_pose_tree.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/distributed_pose_tree.py"
  COMMENT "Copying distributed_pose_tree.py"
)
add_custom_target(python_distributed_pose_tree ALL
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/distributed_pose_tree.py"
)

# Copy config file
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/distributed_pose_tree.yaml"
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/distributed_pose_tree.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/distributed_pose_tree.yaml"
  COMMENT "Copying distributed_pose_tree.yaml"
)
add_custom_target(python_distributed_pose_tree_yaml
  DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/distributed_pose_tree.yaml"
)

add_dependencies(python_distributed_pose_tree python_distributed_pose_tree_yaml)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/distributed_pose_tree.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/distributed_pose_tree.yaml"
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
  add_test(NAME EXAMPLE_PYTHON_DISTRIBUTED_POSE_TREE_TEST
    COMMAND ${Python3_EXECUTABLE} distributed_pose_tree.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_DISTRIBUTED_POSE_TREE_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "day  365"
    FAIL_REGULAR_EXPRESSION "error"
    RESOURCE_LOCK "distributed_ucx"
  )
endif()
