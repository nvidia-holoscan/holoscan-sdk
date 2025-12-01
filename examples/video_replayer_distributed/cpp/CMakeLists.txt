# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
add_executable(video_replayer_distributed
  video_replayer_distributed.cpp
)

target_link_libraries(video_replayer_distributed
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
)

# Copy config file
add_custom_target(video_replayer_distributed_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer_distributed.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "video_replayer_distributed.yaml"
  BYPRODUCTS "video_replayer_distributed.yaml"
)

add_dependencies(video_replayer_distributed video_replayer_distributed_yaml racerx_data)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR})
set_target_properties(video_replayer_distributed PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOSCAN_INSTALL_EXAMPLE_SOURCE)
# Install the source
install(FILES video_replayer_distributed.cpp
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
install(TARGETS video_replayer_distributed
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer_distributed.yaml"
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)

# Testing
if(HOLOSCAN_BUILD_TESTS)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME video_replayer_output)
  set(VALIDATION_FRAMES_DIR ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/video_replayer/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # configure the app with a fixed count of 10 frames (to match previously saved reference data)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/video_replayer_distributed.yaml CONFIG_STRING)
  string(REPLACE "count: 0" "count: 10" CONFIG_STRING ${CONFIG_STRING})
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_video_replayer_distributed_config.yaml)
  file(WRITE ${CONFIG_FILE} ${CONFIG_STRING})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT video_replayer_distributed_test.cpp
    COMMAND patch -u -o video_replayer_distributed_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/video_replayer_distributed.cpp
    ${Holoscan-examples_SOURCE_DIR}/../tests/data/validation_frames/video_replayer/cpp_video_replayer_distributed.patch
  )

  # Create the test executable
  add_executable(video_replayer_distributed_test
  video_replayer_distributed_test.cpp
  )

  target_include_directories(video_replayer_distributed_test
    PRIVATE ${Holoscan-examples_SOURCE_DIR}/../tests)

  target_compile_definitions(video_replayer_distributed_test
    PRIVATE RECORD_OUTPUT RECORDING_DIR="${RECORDING_DIR}"
    PRIVATE SOURCE_VIDEO_BASENAME="${SOURCE_VIDEO_BASENAME}"
  )

  target_link_libraries(video_replayer_distributed_test
    PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::video_stream_replayer
    holoscan::ops::video_stream_recorder
    holoscan::ops::format_converter
  )

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_TEST
    COMMAND bash -c
      "${CMAKE_CURRENT_BINARY_DIR}/video_replayer_distributed_test --config ${CONFIG_FILE} 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'APPLICATION PORT MAPPING' \
        --pass-pattern 'fragment2.holoviz.receivers:0' \
        --pass-pattern 'FRAGMENT fragment1 PORT MAPPING' \
        --pass-pattern 'fragment1.replayer.output:' \
        --pass-pattern 'FRAGMENT fragment2 PORT MAPPING' \
        --fail-pattern 'initialized independent of a parent entity'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_DRIVER_AND_WORKER_TEST
    COMMAND bash -c
      "${CMAKE_CURRENT_BINARY_DIR}/video_replayer_distributed_test --config ${CONFIG_FILE} --driver --worker --fragments=all 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'APPLICATION PORT MAPPING' \
        --pass-pattern 'fragment2.holoviz.receivers:0' \
        --pass-pattern 'FRAGMENT fragment1 PORT MAPPING' \
        --pass-pattern 'fragment1.replayer.output:' \
        --pass-pattern 'FRAGMENT fragment2 PORT MAPPING' \
        --fail-pattern 'initialized independent of a parent entity'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  # repeat the default test, but with HOLOSCAN_UCX_ASYNCHRONOUS=TRUE to set
  # enable_async=true for UcxContext
  # TODO(Holoscan v4.0): remove HOLOSCAN_UCX_ASYNCHRONOUS environment variable and this test case
  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_ASYNCHRONOUS_MODE_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/video_replayer_distributed_test --config ${CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_ASYNCHRONOUS_MODE_TEST PROPERTIES
    ENVIRONMENT "HOLOSCAN_UCX_ASYNCHRONOUS=True"
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
    FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
  )

  # Add a test to check the validity of the frames
  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_RENDER_TEST
      COMMAND ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_RENDER_TEST PROPERTIES
  DEPENDS EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_TEST
  PASS_REGULAR_EXPRESSION "Valid video output!"
  FAIL_REGULAR_EXPRESSION "initialized independent of a parent entity"
  )

  # Create dual window version that will auto-close one window after 30 frames
  # Also set count=100 just to make sure the app will terminate even if window close did not work
  # as expected
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/video_replayer_distributed.yaml CONFIG_STRING)
  string(REPLACE "count: 0" "count: 100" CONFIG_STRING ${CONFIG_STRING})
  string(REPLACE "dual_window: false" "dual_window: true" CONFIG_STRING ${CONFIG_STRING})
  set(DUAL_WINDOW_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_video_replayer_distributed_dual_window_config.yaml)
  file(WRITE ${DUAL_WINDOW_CONFIG_FILE} ${CONFIG_STRING})

  add_test(NAME EXAMPLE_CPP_VIDEO_REPLAYER_DISTRIBUTED_WINDOW_CLOSE_TEST
    COMMAND bash -c
      "${CMAKE_CURRENT_BINARY_DIR}/video_replayer_distributed --config ${DUAL_WINDOW_CONFIG_FILE} --driver --worker --fragments=all 2>&1 | \
      ${Python3_EXECUTABLE} ${Holoscan-examples_SOURCE_DIR}/../scripts/test_pattern_validation.py --strip-quotes \
        --pass-pattern 'HolovizOpAutoClose: compute called 30 times' \
        --pass-pattern 'Terminating fragment fragment1 via stop_execution()' \
        --fail-pattern 'HolovizOpAutoClose: compute called 31 times' \
        --fail-pattern '\\[error\\]'"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
endif()
