# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    ${HOLOSCAN_TOP}/gxf_extensions # TODO(unknown): expose in targets instead
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
    "LD_LIBRARY_PATH=${BIN_DIR}/lib:${BIN_DIR}/gxf_extensions/format_converter:${BIN_DIR}/gxf_extensions/holoviz:${BIN_DIR}/gxf_extensions/stream_playback:$ENV{LD_LIBRARY_PATH}")

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
  core/entity_group.cpp
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

if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Set test properties for CORE_TEST
  # (death tests have to be excluded because it causes an issue with ASAN)
  # https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
  set_tests_properties(CORE_TEST PROPERTIES ENVIRONMENT "GTEST_FILTER=-Application.TestAppHelpOption:Application.TestAppVersionOption")
endif()

# ##################################################################################################
# * pose_tree tests ----------------------------------------------------------------------------------
ConfigureTest(POSE_TREE_TEST
  pose_tree/pose_tree.cpp
  pose_tree/pose_tree_history.cpp
)
target_link_libraries(POSE_TREE_TEST
  PRIVATE
  holoscan::pose_tree
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
  holoscan::ops::ping_tensor_rx
  holoscan::ops::ping_tensor_tx
  holoscan::ops::segmentation_postprocessor
  holoscan::ops::v4l2
  holoscan::ops::video_stream_recorder
  holoscan::ops::video_stream_replayer
)

add_dependencies(OPERATORS_CLASSES_TEST racerx_data)

# ##################################################################################################
# * data logger class unit tests ----------------------------------------------------------------------------------
ConfigureTest(DATA_LOGGER_CLASSES_TEST
  data_loggers/data_logger_classes.cpp
)
target_link_libraries(DATA_LOGGER_CLASSES_TEST
  PRIVATE
  holoscan::data_loggers::basic_console_logger
)

# ##################################################################################################
# * system tests ----------------------------------------------------------------------------------
ConfigureTest(
  SYSTEM_TEST
  system/condition_combiner.cpp
  system/cuda_allocators.cpp
  system/cuda_stream.cpp
  system/cycle.cpp
  system/emit_receive_port_name_test.cpp
  system/env_wrapper.cpp
  system/exception_handling.cpp
  system/demosaic_op_app.cpp
  system/format_converter_op_apps.cpp
  system/jobstatistics_app.cpp
  system/holoviz_op_apps.cpp
  system/loading_gxf_extension.cpp
  system/message_conditions_via_constructor.cpp
  system/multi_receiver_operator_ping_app.cpp
  system/multiple_run_app.cpp
  system/multithreaded_app.cpp
  system/native_async_operator_ping_app.cpp
  system/native_operator_minimal_app.cpp
  system/native_operator_multibroadcasts_app.cpp
  system/native_operator_ping_app.cpp
  system/operator_condition_and_resource_methods.cpp
  system/operator_metadata_apps.cpp
  system/operator_status_test.cpp
  system/ping_rx_op.cpp
  system/ping_tx_op.cpp
  system/receive_as_entity_test.cpp
  system/tensor_compare_op.cpp
  system/video_stream_replayer_op.cpp
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
  holoscan::ops::ping_tensor_rx
  holoscan::ops::ping_tensor_tx
  holoscan::ops::video_stream_replayer
)

if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Exclude tests that are known to fail with ASAN
  set_tests_properties(SYSTEM_TEST PROPERTIES ENVIRONMENT "GTEST_FILTER=-HolovizApps.*")
endif()

# ##################################################################################################
# * utils tests ------------------------------------------------------------------------------------
ConfigureTest(
  UTILS_TEST
  utils/cuda_macros.cpp
  utils/operator_runner_test.cpp
)

# ##################################################################################################
# * system distributed tests ------------------------------------------------------------------------
ConfigureTest(
  SYSTEM_DISTRIBUTED_TEST
  system/distributed/distributed_app.cpp
  system/distributed/distributed_demosaic_op_app.cpp
  system/distributed/distributed_gxf_operator_app.cpp
  system/distributed/holoscan_ucx_ports_env.cpp
  system/distributed/ping_message_rx_op.cpp
  system/distributed/ping_message_tx_op.cpp
  system/distributed/single_tensor_transmit.cpp
  system/distributed/standalone_fragments.cpp
  system/distributed/ucx_message_serialization_ping_app.cpp
  system/env_wrapper.cpp
)
target_link_libraries(SYSTEM_DISTRIBUTED_TEST
  PRIVATE
  holoscan::ops::bayer_demosaic
  holoscan::ops::gxf_codelet
  holoscan::ops::holoviz
  holoscan::ops::ping_tensor_rx
  holoscan::ops::ping_tensor_tx
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

if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Append additional flags to exclude tests that are known to fail with ASAN
  string(APPEND CMAKE_SYSTEM_DISTRIBUTED_TEST_FLAGS
    ";GTEST_FILTER=-DistributedApp.TestTwoMultiInputsOutputsFragmentsApp"
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
  system/distributed/distributed_gxf_operator_app.cpp
  system/distributed/holoscan_ucx_ports_env.cpp
  system/distributed/single_tensor_transmit.cpp
  system/distributed/standalone_fragments.cpp
  system/env_wrapper.cpp
)
target_link_libraries(SYSTEM_DISTRIBUTED_EBS_TEST
  PRIVATE
  holoscan::ops::bayer_demosaic
  holoscan::ops::gxf_codelet
  holoscan::ops::holoviz
  holoscan::ops::ping_tensor_rx
  holoscan::ops::ping_tensor_tx
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
  flow_tracking/entity_passthrough.cpp
  flow_tracking/flow_tracking_cycle.cpp
  flow_tracking/limited_tracking_tests.cpp
)
target_link_libraries(FLOW_TRACKING_TEST
  PRIVATE
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
)
