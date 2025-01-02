# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_dependencies(python_holoviz_geometry racerx_data)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Holoviz Geometry Testing
if(HOLOSCAN_BUILD_TESTS)
  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_holoviz_geometry_output)
  set(VALIDATION_FRAMES_DIR ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/holoviz_geometry/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT holoviz_geometry_test.py
    COMMAND patch -u -o holoviz_geometry_test.py ${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry.py
      ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/holoviz_geometry/python_holoviz_geometry.patch
    COMMAND sed -i "s#RECORDING_DIR#${RECORDING_DIR}#g" holoviz_geometry_test.py
    COMMAND sed -i "s#SOURCE_VIDEO_BASENAME#${SOURCE_VIDEO_BASENAME}#g" holoviz_geometry_test.py
  )

  add_custom_target(python_holoviz_geometry_test ALL
    DEPENDS holoviz_geometry_test.py
  )

  add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_TEST
    COMMAND python3 holoviz_geometry_test.py --count 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Received camera pose:"
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )

    # Add a test to check the validity of the frames
    add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_RENDER_TEST
    COMMAND python3 ${Holoscan-examples_SOURCE_DIR}/../scripts/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_RENDER_TEST PROPERTIES
    DEPENDS EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_TEST
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )
endif()

# Copy Holoviz geometry 3D application
add_custom_target(python_holoviz_geometry_3d ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry_3d.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "holoviz_geometry_3d.py"
  BYPRODUCTS "holoviz_geometry_3d.py"
)

add_dependencies(python_holoviz_geometry_3d racerx_data)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry_3d.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Install the minimal CMakeLists.txt file
install(FILES CMakeLists.min.txt
  RENAME "CMakeLists.txt"
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Holoviz Geometry 3D Testing
if(HOLOSCAN_BUILD_TESTS)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_holoviz_geometry_3d_output)
  set(VALIDATION_FRAMES_DIR ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/holoviz_geometry_3d/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT holoviz_geometry_3d_test.py
    COMMAND patch -u -o holoviz_geometry_3d_test.py ${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry_3d.py
      ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/holoviz_geometry_3d/python_holoviz_geometry_3d.patch
    COMMAND sed -i "s#RECORDING_DIR#${RECORDING_DIR}#g" holoviz_geometry_3d_test.py
    COMMAND sed -i "s#SOURCE_VIDEO_BASENAME#${SOURCE_VIDEO_BASENAME}#g" holoviz_geometry_3d_test.py
  )

  add_custom_target(python_holoviz_geometry_3d_test ALL
    DEPENDS holoviz_geometry_3d_test.py
  )

  add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_3D_TEST
    COMMAND python3 holoviz_geometry_3d_test.py --count 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_3D_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Scheduler finished."
  )

    # Add a test to check the validity of the frames
    add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_3D_RENDER_TEST
    COMMAND python3 ${Holoscan-examples_SOURCE_DIR}/../scripts/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_3D_RENDER_TEST PROPERTIES
    DEPENDS EXAMPLE_PYTHON_HOLOVIZ_GEOMETRY_3D_TEST
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )
endif()

# Copy Holoviz views application
add_custom_target(python_holoviz_views ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_views.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "holoviz_views.py"
  BYPRODUCTS "holoviz_views.py"
)

add_dependencies(python_holoviz_views racerx_data)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/holoviz_views.py"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Holoviz Views Testing
if(HOLOSCAN_BUILD_TESTS)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_holoviz_views_output)
  set(VALIDATION_FRAMES_DIR ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/holoviz_views/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT holoviz_views_test.py
    COMMAND patch -u -o holoviz_views_test.py ${CMAKE_CURRENT_SOURCE_DIR}/holoviz_views.py
      ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/holoviz_views/python_holoviz_views.patch
    COMMAND sed -i "s#RECORDING_DIR#${RECORDING_DIR}#g" holoviz_views_test.py
    COMMAND sed -i "s#SOURCE_VIDEO_BASENAME#${SOURCE_VIDEO_BASENAME}#g" holoviz_views_test.py
  )

  add_custom_target(python_holoviz_views_test ALL
    DEPENDS holoviz_views_test.py
  )

  add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_VIEWS_TEST
    COMMAND python3 holoviz_views_test.py --count 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_VIEWS_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )

    # Add a test to check the validity of the frames
    add_test(NAME EXAMPLE_PYTHON_HOLOVIZ_VIEWS_RENDER_TEST
    COMMAND python3 ${Holoscan-examples_SOURCE_DIR}/../scripts/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_HOLOVIZ_VIEWS_RENDER_TEST PROPERTIES
    DEPENDS EXAMPLE_PYTHON_HOLOVIZ_VIEWS_TEST
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )
endif()
