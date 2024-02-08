# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy video_replayer application file
add_custom_target(python_video_replayer ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "video_replayer.py"
  BYPRODUCTS "video_replayer.py"
)

# Copy config file
add_custom_target(python_video_replayer_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "video_replayer.yaml"
  BYPRODUCTS "video_replayer.yaml"
)

add_dependencies(python_video_replayer python_video_replayer_yaml racerx_data)

# Install the app
install(FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml"
    DESTINATION "${app_relative_dest_path}"
    COMPONENT "holoscan-examples"
)

# Testing
if(HOLOSCAN_BUILD_TESTS)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_video_replayer_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/tests/data/validation_frames/video_replayer/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.yaml CONFIG_STRING)
  string(REPLACE "count: 0" "count: 10" CONFIG_STRING ${CONFIG_STRING})
  string(APPEND CONFIG_STRING "  enable_render_buffer_output: true\n\nrecorder:\n  directory: \"${RECORDING_DIR}\"\n  basename: \"${SOURCE_VIDEO_BASENAME}\"")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/python_video_replayer_config.yaml)
  file(WRITE ${CONFIG_FILE} ${CONFIG_STRING})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT video_replayer_test.py
    PRE_LINK
    COMMAND patch -u -o video_replayer_test.py ${CMAKE_CURRENT_SOURCE_DIR}/video_replayer.py
        ${CMAKE_SOURCE_DIR}/tests/data/validation_frames/video_replayer/python_video_replayer.patch
  )

  add_custom_target(python_video_replayer_test ALL
    DEPENDS "video_replayer_test.py"
  )

  add_test(NAME EXAMPLE_PYTHON_VIDEO_REPLAYER_TEST
    COMMAND python3 video_replayer_test.py --config python_video_replayer_config.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_VIDEO_REPLAYER_TEST PROPERTIES
    DEPENDS "video_replayer_test.py"
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
  )

  # Add a test to check the validity of the frames
  add_test(NAME EXAMPLE_PYTHON_VIDEO_REPLAYER_RENDER_TEST
      COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(EXAMPLE_PYTHON_VIDEO_REPLAYER_RENDER_TEST PROPERTIES
    DEPENDS EXAMPLE_PYTHON_VIDEO_REPLAYER_TEST
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()
