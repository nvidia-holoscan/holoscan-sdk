# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_executable(video_replayer
  video_replayer.cpp
)

target_link_libraries(video_replayer
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
)

# Copy config file
add_custom_target(video_replayer_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "video_replayer.yaml"
  BYPRODUCTS "video_replayer.yaml"
)

add_dependencies(video_replayer video_replayer_yaml racerx_data)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(video_replayer PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES video_replayer.cpp
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
install(TARGETS video_replayer
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml"
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME video_replayer_output)
  set(VALIDATION_FRAMES_DIR ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/video_replayer/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})


  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml CONFIG_STRING)
  string(REPLACE "count: 0" "count: 10" CONFIG_STRING "${CONFIG_STRING}")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_video_replayer_config.yaml)
  file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

  # now also write a second config with dual_window set to true
  string(REPLACE "dual_window: false" "dual_window: true" CONFIG_STRING "${CONFIG_STRING}")
  set(DUAL_WINDOW_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_video_replayer_dual_config.yaml)
  file(WRITE ${DUAL_WINDOW_CONFIG_FILE} "${CONFIG_STRING}")

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT video_replayer_test.cpp
    PRE_LINK
    COMMAND patch -u -o video_replayer_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.cpp
      ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/video_replayer/cpp_video_replayer.patch
  )

  # Create the test executable
  add_executable(video_replayer_test
    video_replayer_test.cpp
  )

  target_include_directories(video_replayer_test
    PRIVATE ${Holoscan-examples_SOURCE_DIR}/../tests)

  target_compile_definitions(video_replayer_test
    PRIVATE RECORD_OUTPUT RECORDING_DIR="${RECORDING_DIR}"
    PRIVATE SOURCE_VIDEO_BASENAME="${SOURCE_VIDEO_BASENAME}"
  )

  target_link_libraries(video_replayer_test
    PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::video_stream_replayer
    holoscan::ops::video_stream_recorder
    holoscan::ops::format_converter
  )

  # Add the test and make sure it runs
  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/video_replayer_test ${CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_VIDEO_REPLAYER_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )

  # Add a test to check the validity of the frames
  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_RENDER_TEST
      COMMAND python3 ${Holoscan-examples_SOURCE_DIR}/../scripts/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_CPP_VIDEO_REPLAYER_RENDER_TEST PROPERTIES
  DEPENDS EXAMPLE_CPP_VIDEO_REPLAYER_TEST
  PASS_REGULAR_EXPRESSION "Valid video output!"
  )

  # Add the dual-window test and make sure it runs
  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_DUAL_WINDOW_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/video_replayer ${DUAL_WINDOW_CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(EXAMPLE_CPP_VIDEO_REPLAYER_DUAL_WINDOW_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )
endif()
