# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy Holoviz geometry application
add_custom_target(python_holoviz_geometry ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "holoviz_geometry.py"
  BYPRODUCTS "holoviz_geometry.py"
)

add_dependencies(python_holoviz_geometry endoscopy_data)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_TEST
    COMMAND python3 holoviz_geometry.py --count 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_TEST PROPERTIES
  PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.")
endif()

# Copy Holoviz geometry 3D application
add_custom_target(python_holoviz_geometry_3d ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry_3d.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "holoviz_geometry_3d.py"
  BYPRODUCTS "holoviz_geometry_3d.py"
)

add_dependencies(python_holoviz_geometry_3d endoscopy_data)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry_3d.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_3D_TEST
    COMMAND python3 holoviz_geometry_3d.py --count 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_3D_TEST PROPERTIES
  PASS_REGULAR_EXPRESSION "Scheduler finished.")
endif()

# Copy Holoviz views application
add_custom_target(python_holoviz_views ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_views.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "holoviz_views.py"
  BYPRODUCTS "holoviz_views.py"
)

add_dependencies(python_holoviz_views endoscopy_data)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_views.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)
  add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_VIEWS_TEST
    COMMAND python3 holoviz_views.py --count 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_VIEWS_TEST PROPERTIES
  PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.")
endif()
