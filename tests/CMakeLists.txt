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
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::ping_rx
    holoscan::ops::ping_tx
    GTest::gmock_main
    GTest::gtest_main
  )

  # Run the test from the main bin directory to access data as needed
  add_test(NAME ${CMAKE_TEST_NAME} COMMAND ${CMAKE_TEST_NAME} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_property(TEST ${CMAKE_TEST_NAME} PROPERTY ENVIRONMENT
    "LD_LIBRARY_PATH=${BIN_DIR}/lib:${BIN_DIR}/gxf_extensions/aja:${BIN_DIR}/gxf_extensions/format_converter:${BIN_DIR}/gxf_extensions/holoviz:${BIN_DIR}/gxf_extensions/stream_playback:$ENV{LD_LIBRARY_PATH}")

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
  core/app_driver.cpp
  core/application.cpp
  core/arg.cpp
  core/argument_setter.cpp
  core/cli_options.cpp
  core/component.cpp
  core/component_spec.cpp
  core/condition.cpp
  core/condition_classes.cpp
  core/config.cpp
  core/data_exporter.cpp
  core/dataflow_tracker.cpp
  core/fragment.cpp
  core/fragment_allocation.cpp
  core/io_spec.cpp
  core/logger.cpp
  core/message.cpp
  core/metadata.cpp
  core/operator_spec.cpp
  core/parameter.cpp
  core/resource.cpp
  core/resource_classes.cpp
  core/scheduler_classes.cpp
  core/system_resource_manager.cpp
 )

# ##################################################################################################
# * codecs tests ----------------------------------------------------------------------------------
ConfigureTest(CODECS_TEST
  codecs/codec_registry.cpp
  codecs/codecs.cpp
  codecs/mock_allocator.cpp
  codecs/mock_serialization_buffer.cpp
)

# ##################################################################################################
# * operator classes tests ----------------------------------------------------------------------------------
ConfigureTest(OPERATORS_CLASSES_TEST
  operators/operator_classes.cpp
)
target_link_libraries(OPERATORS_CLASSES_TEST
  PRIVATE
  holoscan::ops::async_ping_rx
  holoscan::ops::async_ping_tx
  holoscan::ops::bayer_demosaic
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  holoscan::ops::inference
  holoscan::ops::inference_processor
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
  holoscan::ops::segmentation_postprocessor
  holoscan::ops::v4l2
  holoscan::ops::video_stream_recorder
  holoscan::ops::video_stream_replayer
)

if(HOLOSCAN_BUILD_AJA)
  target_link_libraries(OPERATORS_CLASSES_TEST PRIVATE holoscan::ops::aja)
  target_compile_definitions(OPERATORS_CLASSES_TEST PRIVATE HOLOSCAN_BUILD_AJA)
endif()

add_dependencies(OPERATORS_CLASSES_TEST racerx_data)

# ##################################################################################################
# * system tests ----------------------------------------------------------------------------------
ConfigureTest(
  SYSTEM_TEST
  system/cycle.cpp
  system/env_wrapper.cpp
  system/exception_handling.cpp
  system/demosaic_op_app.cpp
  system/format_converter_op_apps.cpp
  system/jobstatistics_app.cpp
  system/holoviz_op_apps.cpp
  system/loading_gxf_extension.cpp
  system/multi_receiver_operator_ping_app.cpp
  system/multithreaded_app.cpp
  system/native_async_operator_ping_app.cpp
  system/native_operator_minimal_app.cpp
  system/native_operator_multibroadcasts_app.cpp
  system/native_operator_ping_app.cpp
  system/operator_metadata_apps.cpp
  system/ping_rx_op.cpp
  system/ping_tensor_rx_op.cpp
  system/ping_tensor_tx_op.cpp
  system/ping_tx_op.cpp
  system/tensor_compare_op.cpp
)
target_link_libraries(SYSTEM_TEST
  PRIVATE
  holoscan::ops::async_ping_rx
  holoscan::ops::async_ping_tx
  holoscan::ops::bayer_demosaic
  holoscan::ops::format_converter
  holoscan::ops::gxf_codelet
  holoscan::ops::holoviz
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
)

ConfigureTest(
  SYSTEM_DISTRIBUTED_TEST
  system/distributed/distributed_app.cpp
  system/distributed/distributed_demosaic_op_app.cpp
  system/distributed/holoscan_ucx_ports_env.cpp
  system/distributed/ping_message_rx_op.cpp
  system/distributed/ping_message_tx_op.cpp
  system/distributed/standalone_fragments.cpp
  system/distributed/ucx_message_serialization_ping_app.cpp
  system/env_wrapper.cpp
  system/ping_tensor_rx_op.cpp
  system/ping_tensor_tx_op.cpp
)
target_link_libraries(SYSTEM_DISTRIBUTED_TEST
  PRIVATE
  holoscan::ops::bayer_demosaic
  holoscan::ops::holoviz
)

# set environment variables used by distributed applications in the tests
# - HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=3000    : Set deadlock timeout for distributed app
# - HOLOSCAN_MAX_DURATION_MS=3000             : Set max duration for distributed app
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(CMAKE_SYSTEM_DISTRIBUTED_TEST_FLAGS
"\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=6000;\
HOLOSCAN_MAX_DURATION_MS=6000;\
HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=multi_thread\
"
  )
else()
  set(CMAKE_SYSTEM_DISTRIBUTED_TEST_FLAGS
"\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=3000;\
HOLOSCAN_MAX_DURATION_MS=3000;\
HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=multi_thread\
"
  )
endif()

set_tests_properties(
  SYSTEM_DISTRIBUTED_TEST PROPERTIES ENVIRONMENT "${CMAKE_SYSTEM_DISTRIBUTED_TEST_FLAGS}"
)

# Repeat distributed tests, but setting environment variables to use the event-based scheduler
# (omitting UcxMessageSerialization tests in this case to reduce overall test time)
ConfigureTest(
  SYSTEM_DISTRIBUTED_EBS_TEST
  system/distributed/distributed_app.cpp
  system/distributed/distributed_demosaic_op_app.cpp
  system/distributed/holoscan_ucx_ports_env.cpp
  system/distributed/standalone_fragments.cpp
  system/env_wrapper.cpp
  system/ping_tensor_rx_op.cpp
  system/ping_tensor_tx_op.cpp
)
target_link_libraries(SYSTEM_DISTRIBUTED_EBS_TEST
  PRIVATE
  holoscan::ops::bayer_demosaic
  holoscan::ops::holoviz
)

  set(CMAKE_SYSTEM_DISTRIBUTED_EBS_TEST_FLAGS
"\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=5000;\
HOLOSCAN_MAX_DURATION_MS=5000;\
HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=event_based\
"
  )
set_tests_properties(
  SYSTEM_DISTRIBUTED_EBS_TEST PROPERTIES ENVIRONMENT "${CMAKE_SYSTEM_DISTRIBUTED_EBS_TEST_FLAGS}")


# ##################################################################################################
# * stress tests ----------------------------------------------------------------------------------
ConfigureTest(
  STRESS_TEST
  stress/ping_multi_port_test.cpp
)

# #######
ConfigureTest(SEGMENTATION_POSTPROCESSOR_TEST
  operators/segmentation_postprocessor/test_postprocessor.cpp
)
target_link_libraries(SEGMENTATION_POSTPROCESSOR_TEST
  PRIVATE
    holoscan::ops::segmentation_postprocessor
)

# #######
ConfigureTest(HOLOINFER_TEST
  holoinfer/inference/test_core.cpp
  holoinfer/inference/test_inference.cpp
  holoinfer/inference/test_parameters.cpp
  holoinfer/inference/test_core.hpp
  holoinfer/inference/test_infer_settings.hpp
  holoinfer/processing/test_core.cpp
  holoinfer/processing/test_core.hpp
  holoinfer/processing/test_parameters.cpp
  holoinfer/holoinfer_test_driver.cpp
)
target_include_directories(HOLOINFER_TEST
  PRIVATE
    ${CMAKE_SOURCE_DIR}/modules/holoinfer/src/include
)

target_link_libraries(HOLOINFER_TEST
  PRIVATE
    holoinfer
)

# ##################################################################################################
# * Flow Tracking tests ----------------------------------------------------------------------------------
ConfigureTest(
  FLOW_TRACKING_TEST
  flow_tracking/flow_tracking_cycle.cpp
  flow_tracking/entity_passthrough.cpp
)
target_link_libraries(FLOW_TRACKING_TEST
  PRIVATE
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
)
