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

# Create example
add_executable(holoviz_geometry
  holoviz_geometry.cpp
)

target_link_libraries(holoviz_geometry
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::video_stream_replayer
)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(holoviz_geometry PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES holoviz_geometry.cpp
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

# Install the app
install(TARGETS holoviz_geometry
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME holoviz_geometry_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/tests/data/validation_frames/holoviz_geometry/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT holoviz_geometry_test.cpp
    PRE_LINK
    COMMAND patch -u -o holoviz_geometry_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/holoviz_geometry.cpp
            ${CMAKE_SOURCE_DIR}/tests/data/validation_frames/holoviz_geometry/cpp_holoviz_geometry.patch
  )

  # Create the test executable
  add_executable(holoviz_geometry_test
                 holoviz_geometry_test.cpp
  )

  target_include_directories(holoviz_geometry_test
    PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  target_compile_definitions(holoviz_geometry_test
    PRIVATE RECORD_OUTPUT RECORDING_DIR="${RECORDING_DIR}"
    PRIVATE SOURCE_VIDEO_BASENAME="${SOURCE_VIDEO_BASENAME}"
  )

  target_link_libraries(holoviz_geometry_test
    PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::video_stream_replayer
    holoscan::ops::video_stream_recorder
    holoscan::ops::format_converter
  )

  # Add the test and make sure it runs
  add_test(NAME EXAMPLE_CPP_HOLOVIZ_GEOMETRY_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/holoviz_geometry_test --count 10
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_HOLOVIZ_GEOMETRY_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Received camera pose:"
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )

  # Add a test to check the validity of the frames
  add_test(NAME EXAMPLE_CPP_HOLOVIZ_GEOMETRY_RENDER_TEST
      COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_CPP_HOLOVIZ_GEOMETRY_RENDER_TEST PROPERTIES
    DEPENDS EXAMPLE_CPP_HOLOVIZ_GEOMETRY_TEST
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()
