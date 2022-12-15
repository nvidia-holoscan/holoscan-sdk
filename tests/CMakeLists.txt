# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest CMAKE_TEST_NAME)
  
  add_executable(${CMAKE_TEST_NAME} ${ARGN})
  
  set(BIN_DIR ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR})

  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${BIN_DIR}/gtests>"
  )

  target_include_directories(${CMAKE_TEST_NAME}
    PRIVATE
    ${HOLOSCAN_TOP}/gxf_extensions # TODO: expose in targets instead
  )

  target_link_libraries(${CMAKE_TEST_NAME}
    PRIVATE
    ${HOLOSCAN_PACKAGE_NAME}
    GTest::gmock_main
    GTest::gtest_main
  )

  # Run the test from the main bin directory to access data as needed
  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_property(TEST ${CMAKE_TEST_NAME} PROPERTY ENVIRONMENT
    "LD_LIBRARY_PATH=${BIN_DIR}/lib:${BIN_DIR}/gxf_extensions/aja:${BIN_DIR}/gxf_extensions/custom_lstm_inference:${BIN_DIR}/gxf_extensions/format_converter:${BIN_DIR}/gxf_extensions/holoviz:${BIN_DIR}/gxf_extensions/segmentation_postprocessor:${BIN_DIR}/gxf_extensions/stream_playback:${BIN_DIR}/gxf_extensions/tensor_rt:${BIN_DIR}/gxf_extensions/tool_tracking_postprocessor:${BIN_DIR}/gxf_extensions/visualizer_tool_tracking:$ENV{LD_LIBRARY_PATH}")

  install(
    TARGETS ${CMAKE_TEST_NAME}
    COMPONENT holoscan-testing
    DESTINATION bin/gtests/libholoscan
    EXCLUDE_FROM_ALL
  )
endfunction()

# ##################################################################################################
# * core tests ----------------------------------------------------------------------------------
ConfigureTest(
  CORE_TEST
  core/application.cpp
  core/arg.cpp
  core/argument_setter.cpp
  core/component.cpp
  core/component_spec.cpp
  core/condition.cpp
  core/condition_classes.cpp
  core/config.cpp
  core/fragment.cpp
  core/io_spec.cpp
  core/logger.cpp
  core/operator_classes.cpp
  core/parameter.cpp
  core/resource.cpp
  core/resource_classes.cpp
  config.hpp
  utils.hpp
)

# #######
ConfigureTest(SEGMENTATION_POSTPROCESSOR_TEST
  gxf_extensions/segmentation_postprocessor/test_postprocessor.cpp
)
target_link_libraries(SEGMENTATION_POSTPROCESSOR_TEST
  PRIVATE
    CUDA::cudart
    segmentation_postprocessor
)

# #######
ConfigureTest(HOLOINFER_TEST
  holoinfer/multiai_tests.cpp
  holoinfer/multiai_tests.hpp
  holoinfer/test_infer_settings.hpp
)
target_include_directories(HOLOINFER_TEST
  PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/holoinfer/src/include
)
target_link_libraries(HOLOINFER_TEST
  PRIVATE
    holoinfer
)
