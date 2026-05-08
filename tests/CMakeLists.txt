# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# * Fetch dependencies -----------------------------------------------------------------------------

# Reuse stb from holoviz if available, otherwise fetch it
include(FetchContent)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
  FetchContent_Declare(
      stb
      GIT_REPOSITORY https://github.com/nothings/stb.git
      GIT_TAG af1a5bc352164740c1cc1354942b1c6b72eacb8a
      GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(stb)
endif()

# ##################################################################################################
# * compiler function -----------------------------------------------------------------------------

# This function takes in a test name and test source and handles setting all of the associated
# properties and linking to build the test
function(ConfigureTest CMAKE_TEST_NAME)

  add_executable(${CMAKE_TEST_NAME} ${ARGN})

  set(BIN_DIR ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR})

  set_target_properties(
    ${CMAKE_TEST_NAME}
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${BIN_DIR}/gtests>"
      # Disable clang-tidy for test code
      CXX_CLANG_TIDY ""
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
    "LD_LIBRARY_PATH=${BIN_DIR}/lib:${BIN_DIR}/gxf_extensions:${BIN_DIR}/gxf_extensions/format_converter:${BIN_DIR}/gxf_extensions/holoviz:${BIN_DIR}/gxf_extensions/stream_playback:$ENV{LD_LIBRARY_PATH};HOLOSCAN_LIB_PATH=${BIN_DIR}/gxf_extensions:${BIN_DIR}/lib:$ENV{HOLOSCAN_LIB_PATH}")

  install(
    TARGETS ${CMAKE_TEST_NAME}
    COMPONENT holoscan-testing
    DESTINATION bin/gtests/libholoscan
    EXCLUDE_FROM_ALL
  )
endfunction()

# Copy test data files into the build tree so that tests can reference them via
# paths relative to their WORKING_DIRECTORY (CMAKE_BINARY_DIR).
file(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/operators/inference/inference.yaml
  ${CMAKE_CURRENT_SOURCE_DIR}/operators/inference/models
  DESTINATION ${CMAKE_BINARY_DIR}/tests/operators/inference
)
file(COPY
  ${CMAKE_CURRENT_SOURCE_DIR}/holoinfer/test_models
  DESTINATION ${CMAKE_BINARY_DIR}/tests/holoinfer
)

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
  core/flow_graph_impl.cpp
  core/fragment.cpp
  core/fragment_allocation.cpp
  core/io_spec.cpp
  core/logger.cpp
  core/message.cpp
  core/metadata.cpp
  core/network_context_classes.cpp
  core/operator_spec.cpp
  core/parameter.cpp
  core/resource.cpp
  core/resource_classes.cpp
  core/scheduler_classes.cpp
  core/system_resource_manager.cpp
)
add_dependencies(CORE_TEST gxf_ucx_holoscan)

# ##################################################################################################
# * pubsub runtime tests ---------------------------------------------------------------------------
ConfigureTest(PUBSUB_RUNTIME_TEST
  pubsub/runtime_condition_classes.cpp
)
target_link_libraries(PUBSUB_RUNTIME_TEST
  PRIVATE
    holoscan::pubsub_runtime
)

# ##################################################################################################
# * pubsub in-memory tests -------------------------------------------------------------------------
ConfigureTest(PUBSUB_IN_MEMORY_TEST
  pubsub/in_memory_network_context_classes.cpp
  pubsub/in_memory_pubsub_session.cpp
)
add_dependencies(PUBSUB_IN_MEMORY_TEST gxf_ucx_holoscan)
target_link_libraries(PUBSUB_IN_MEMORY_TEST
  PRIVATE
    holoscan::pubsub_in_memory
)

# ##################################################################################################
# * pubsub FastDDS system tests --------------------------------------------------------------------
if(TARGET holoscan::pubsub_fastdds)
  ConfigureTest(PUBSUB_FASTDDS_BACKEND_TEST
    system/pubsub/fastdds/fastdds_backend_test.cpp
  )
  target_include_directories(PUBSUB_FASTDDS_BACKEND_TEST
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/system/pubsub/fastdds
  )
  target_link_libraries(PUBSUB_FASTDDS_BACKEND_TEST
    PRIVATE
      holoscan::pubsub_fastdds
  )

  ConfigureTest(PUBSUB_FASTDDS_SYSTEM_TEST
    system/pubsub/fastdds/pubsub_loopback_test.cpp
  )
  target_link_libraries(PUBSUB_FASTDDS_SYSTEM_TEST
    PRIVATE
      holoscan::pubsub_fastdds
  )

  ConfigureTest(PUBSUB_FASTDDS_NATIVE_BUFFER_DATA_PATH_TEST
    system/pubsub/fastdds/native_buffer_data_path_test.cpp
  )
  target_include_directories(PUBSUB_FASTDDS_NATIVE_BUFFER_DATA_PATH_TEST
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/system/pubsub/fastdds
  )
  target_link_libraries(PUBSUB_FASTDDS_NATIVE_BUFFER_DATA_PATH_TEST
    PRIVATE
      holoscan::pubsub_fastdds
  )

  ConfigureTest(PUBSUB_FASTDDS_NATIVE_BUFFER_INTEGRATION_TEST
    system/pubsub/fastdds/native_buffer_integration_test.cpp
  )
  target_link_libraries(PUBSUB_FASTDDS_NATIVE_BUFFER_INTEGRATION_TEST
    PRIVATE
      holoscan::pubsub_fastdds
  )

  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_executable(PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP
    system/pubsub/fastdds/native_buffer_ipc_app.cpp
  )
  set_target_properties(
    PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/gtests>"
      CXX_CLANG_TIDY ""
  )
  target_include_directories(PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP
    PRIVATE
      ${HOLOSCAN_TOP}/gxf_extensions
  )
  target_link_libraries(PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP
    PRIVATE
      holoscan::pubsub_fastdds
  )
  install(
    TARGETS PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP
    COMPONENT holoscan-testing
    DESTINATION bin/gtests/libholoscan
    EXCLUDE_FROM_ALL
  )

  add_test(
    NAME PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_TEST
    COMMAND bash -c
      "bash ${CMAKE_CURRENT_SOURCE_DIR}/system/pubsub/fastdds/test_native_buffer_ipc.sh \
       $<TARGET_FILE:PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP> 5 2>&1 | \
       ${Python3_EXECUTABLE} ${HOLOSCAN_TOP}/scripts/test_pattern_validation.py \
         --pass-pattern 'PASS: received 5/5 messages, 0 mismatches' \
         --pass-pattern 'rx: verified message 5 OK' \
         --fail-pattern 'TIMEOUT:' \
         --fail-pattern 'pending exports still outstanding' \
         --fail-pattern 'GXF_PUBSUB_DESCRIPTOR_OPEN_FAIL' \
         --fail-pattern 'rx: no Tensor in entity' \
         --fail-pattern 'rx: data mismatch'"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_TEST PROPERTIES TIMEOUT 60)

  add_test(
    NAME PUBSUB_FASTDDS_NATIVE_BUFFER_FALLBACK_TEST
    COMMAND ${CMAKE_COMMAND} -E env HOLOSCAN_LOG_LEVEL=DEBUG HOLOSCAN_EXECUTOR_LOG_LEVEL=DEBUG bash -c
      "bash ${CMAKE_CURRENT_SOURCE_DIR}/system/pubsub/fastdds/test_native_buffer_fallback.sh \
       $<TARGET_FILE:PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP> 5 2>&1 | \
       ${Python3_EXECUTABLE} ${HOLOSCAN_TOP}/scripts/test_pattern_validation.py \
         --pass-pattern 'PASS\\[fallback\\]: received 5/5 messages, 0 mismatches' \
         --pass-pattern 'PubSubContext: receiving serialized byte payload' \
         --fail-pattern 'TIMEOUT:' \
         --fail-pattern 'pending exports still outstanding' \
         --fail-pattern 'GXF_PUBSUB_DESCRIPTOR_OPEN_FAIL' \
         --fail-pattern 'PubSubContext: receiving native descriptor payload' \
         --fail-pattern 'rx: data mismatch'"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(PUBSUB_FASTDDS_NATIVE_BUFFER_FALLBACK_TEST PROPERTIES TIMEOUT 60)

  add_test(
    NAME PUBSUB_FASTDDS_NATIVE_BUFFER_MIXED_TEST
    COMMAND ${CMAKE_COMMAND} -E env HOLOSCAN_LOG_LEVEL=DEBUG HOLOSCAN_EXECUTOR_LOG_LEVEL=DEBUG bash -c
      "bash ${CMAKE_CURRENT_SOURCE_DIR}/system/pubsub/fastdds/test_native_buffer_mixed.sh \
       $<TARGET_FILE:PUBSUB_FASTDDS_NATIVE_BUFFER_IPC_APP> 5 2>&1 | \
       ${Python3_EXECUTABLE} ${HOLOSCAN_TOP}/scripts/test_pattern_validation.py \
         --pass-pattern 'PASS\\[native\\]: received 5/5 messages, 0 mismatches' \
         --pass-pattern 'PASS\\[byte\\]: received 5/5 messages, 0 mismatches' \
         --pass-pattern 'PubSubContext: receiving native descriptor payload' \
         --pass-pattern 'PubSubContext: receiving serialized byte payload' \
         --fail-pattern 'TIMEOUT:' \
         --fail-pattern 'pending exports still outstanding' \
         --fail-pattern 'GXF_PUBSUB_DESCRIPTOR_OPEN_FAIL' \
         --fail-pattern 'rx: data mismatch'"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(PUBSUB_FASTDDS_NATIVE_BUFFER_MIXED_TEST PROPERTIES TIMEOUT 60)
endif()

# ##################################################################################################
# * MatX interoperability tests ---------------------------------------------------------------------
#
# Keep this as a separate test binary so we don't force CUDA compilation for the full CORE_TEST.
if(CMAKE_CUDA_COMPILER)
  ConfigureTest(
    MATX_INTEROP_TEST
    core/matx_tensor_interop.cu
  )
  target_link_libraries(MATX_INTEROP_TEST PRIVATE matx::matx)

  # MatX custom allocator adapter tests
  ConfigureTest(
    MATX_ALLOCATOR_TEST
    core/matx_allocator_test.cu
  )
  target_link_libraries(MATX_ALLOCATOR_TEST PRIVATE matx::matx)
endif()

if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Set test properties for CORE_TEST
  # (death tests have to be excluded because it causes an issue with ASAN)
  # https://github.com/google/googletest/blob/main/docs/advanced.md#death-tests-and-threads
  set_tests_properties(CORE_TEST PROPERTIES ENVIRONMENT "GTEST_FILTER=-Application.TestAppHelpOption:Application.TestAppVersionOption")
endif()

# ##################################################################################################
# * gpu_resident tests ----------------------------------------------------------------------------------
ConfigureTest(GPU_RESIDENT_TEST
  gpu_resident/gpu_resident_fragment.cpp
  gpu_resident/error_conditions.cpp
  gpu_resident/data_ready_handler.cpp
  gpu_resident/device_ptr_connection.cpp
  gpu_resident/multi_io.cpp
  gpu_resident/test_kernels.cu
  gpu_resident/multi_io_test_kernels.cu
  system/env_wrapper.cpp
)

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
  operators/test_video_io_operators.cpp
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
  holoscan::ops::video_io
  holoscan::ops::video_stream_recorder
  holoscan::ops::video_stream_replayer
)

add_dependencies(OPERATORS_CLASSES_TEST racerx_data)

# ##################################################################################################
# * inference operator tests -----------------------------------------------------------------------
ConfigureTest(INFERENCE_OPERATOR_TEST
  operators/inference/test_inference_operator.cpp
)
target_link_libraries(INFERENCE_OPERATOR_TEST
  PRIVATE
  holoscan::ops::inference
  holoscan::holoinfer_utils
  $<TARGET_NAME_IF_EXISTS:holoscan::infer::torch>
)

# ##################################################################################################
# * ping_rx tests ----------------------------------------------------------------------------------
ConfigureTest(PING_RX_TEST
  operators/test_ping_rx.cpp
)
target_link_libraries(PING_RX_TEST
  PRIVATE
  holoscan::ops::ping_rx
)

# ##################################################################################################
# * ping_rx tests ----------------------------------------------------------------------------------
ConfigureTest(PING_TX_TEST
  operators/test_ping_tx.cpp
)
target_link_libraries(PING_TX_TEST
  PRIVATE
  holoscan::ops::ping_tx
)

# ##################################################################################################
# * data logger class unit tests ----------------------------------------------------------------------------------
ConfigureTest(DATA_LOGGER_CLASSES_TEST
  data_loggers/data_logger_classes.cpp
)
target_link_libraries(DATA_LOGGER_CLASSES_TEST
  PRIVATE
  holoscan::data_loggers::async_console_logger
  holoscan::data_loggers::basic_console_logger
)

# ##################################################################################################
# * in-memory pubsub system tests (full Holoscan application) -------------------------------------
ConfigureTest(IN_MEMORY_PUBSUB_SYSTEM_TEST
  system/pubsub/pubsub_loopback_test.cpp
  system/pubsub/pubsub_fault_injection_test.cpp
  system/env_wrapper.cpp
)
target_link_libraries(IN_MEMORY_PUBSUB_SYSTEM_TEST
  PRIVATE
    holoscan::pubsub_in_memory
    holoscan::pubsub_runtime
)

# ##################################################################################################
# * system tests ----------------------------------------------------------------------------------
ConfigureTest(
  SYSTEM_TEST
  system/async_data_logger_shutdown_test.cpp
  system/condition_combiner.cpp
  system/cuda_allocators.cpp
  system/cuda_green_context.cpp
  system/cuda_stream.cpp
  system/cuda_stream_alloc_in_start.cpp
  system/cuda_stream_condition.cpp
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
  system/multi_receiver_queue_policy_test.cpp
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
  system/realtime_ping_app.cpp
  system/queue_size_warning.cpp
  system/receive_as_entity_test.cpp
  system/subgraph_apps.cpp
  system/tensor_compare_op.cpp
  system/timestamp_emit_receive.cpp
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
add_dependencies(SYSTEM_TEST gxf_ucx_holoscan)

if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Exclude tests that are known to fail with ASAN
  set_tests_properties(SYSTEM_TEST PROPERTIES ENVIRONMENT "GTEST_FILTER=-HolovizApps.*:RealtimePingFirstInFirstOutApp.*:RealtimePingRoundRobinApp.*:RealtimePingDeadlineApp.*")
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
# *
# * SYSTEM_DISTRIBUTED_EBS_TEST: run comprehensive tests with the EventBasedScheduler
# * SYSTEM_DISTRIBUTED_MTS_TEST: repeat a key subset of tests with the legacy MultiThreadScheduler
# *
# * The full set of files under the distributed folder should be included under
# * SYSTEM_DISTRIBUTED_EBS_TEST, which uses the event-based scheduler.
# *
# * For time efficiency, a more minimal set of tests (distributed_app.cpp) is used to test
# * various distributed applications with the Multi-Thread scheduler.
ConfigureTest(
  SYSTEM_DISTRIBUTED_EBS_TEST
  system/distributed/async_data_logger_distributed_shutdown_test.cpp
  system/distributed/distributed_app.cpp
  system/distributed/distributed_app_scheduler_test.cpp
  system/distributed/distributed_demosaic_op_app.cpp
  system/distributed/distributed_gxf_operator_app.cpp
  system/distributed/holoscan_ucx_ports_env.cpp
  system/distributed/ping_message_rx_op.cpp
  system/distributed/ping_message_tx_op.cpp
  system/distributed/pose_tree_ucx_snapshot_delta_ordering_test.cpp
  system/distributed/single_tensor_transmit.cpp
  system/distributed/standalone_fragments.cpp
  system/distributed/ucx_message_serialization_ping_app.cpp
  system/env_wrapper.cpp
)
target_link_libraries(SYSTEM_DISTRIBUTED_EBS_TEST
  PRIVATE
  holoscan::pose_tree
  holoscan::ops::bayer_demosaic
  holoscan::ops::gxf_codelet
  holoscan::ops::holoviz
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
  holoscan::ops::ping_tensor_rx
  holoscan::ops::ping_tensor_tx
  ucxx::ucxx
)

if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(CMAKE_SYSTEM_DISTRIBUTED_EBS_TEST_FLAGS
"\
HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT=6000;\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=1000;\
HOLOSCAN_MAX_DURATION_MS=6000;\
HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=event_based\
"
  )
else()
  set(CMAKE_SYSTEM_DISTRIBUTED_EBS_TEST_FLAGS
"\
HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT=3000;\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=1000;\
HOLOSCAN_MAX_DURATION_MS=3000;\
HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=event_based\
"
  )
endif()

set_tests_properties(
  SYSTEM_DISTRIBUTED_EBS_TEST PROPERTIES ENVIRONMENT "${CMAKE_SYSTEM_DISTRIBUTED_EBS_TEST_FLAGS}")

# Repeat distributed tests, but setting environment variables to use the multi-thread scheduler
# for a key subset of the distributed tests.
ConfigureTest(
  SYSTEM_DISTRIBUTED_MTS_TEST
  system/distributed/distributed_app.cpp
  system/env_wrapper.cpp
)
target_link_libraries(SYSTEM_DISTRIBUTED_MTS_TEST
  PRIVATE
  holoscan::ops::ping_tensor_rx
  holoscan::ops::ping_tensor_tx
)

# set environment variables used by distributed applications in the tests
# - HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=3000    : Set deadlock timeout for distributed app
# - HOLOSCAN_MAX_DURATION_MS=3000             : Set max duration for distributed app
if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(CMAKE_SYSTEM_DISTRIBUTED_MTS_TEST_FLAGS
"\
HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT=6000;\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=1000;\
HOLOSCAN_MAX_DURATION_MS=6000;\
HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=multi_thread\
"
  )
else()
  set(CMAKE_SYSTEM_DISTRIBUTED_MTS_TEST_FLAGS
"\
HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT=3000;\
HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT=1000;\
HOLOSCAN_MAX_DURATION_MS=3000;\
HOLOSCAN_DISTRIBUTED_APP_SCHEDULER=multi_thread\
"
  )
endif()

if(HOLOSCAN_ENABLE_GOOGLE_SANITIZER)
  # Append additional flags to exclude tests that are known to fail with ASAN
  string(APPEND CMAKE_SYSTEM_DISTRIBUTED_MTS_TEST_FLAGS
    ";GTEST_FILTER=-DistributedApp.TestTwoMultiInputsOutputsFragmentsApp"
  )
endif()

set_tests_properties(
  SYSTEM_DISTRIBUTED_MTS_TEST PROPERTIES ENVIRONMENT "${CMAKE_SYSTEM_DISTRIBUTED_MTS_TEST_FLAGS}"
)

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

# ##################################################################################################
# * bayer_demosaic_gpu_resident tests --------------------------------------------------------------
ConfigureTest(BAYER_DEMOSAIC_GPU_RESIDENT_TEST
  operators/bayer_demosaic_gpu_resident/test_bayer_demosaic_gpu_resident.cpp
)
target_include_directories(BAYER_DEMOSAIC_GPU_RESIDENT_TEST
  PRIVATE
    ${stb_SOURCE_DIR}
)
target_link_libraries(BAYER_DEMOSAIC_GPU_RESIDENT_TEST
  PRIVATE
    holoscan::ops::bayer_demosaic_gpu_resident
    CUDA::nppicc
)

# ##################################################################################################
# * holoinfer tests — registered as individual CTest entries via gtest_discover_tests
include(GoogleTest)

add_executable(HOLOINFER_TEST
  holoinfer/inference/test_core.cpp
  holoinfer/inference/test_inference.cpp
  holoinfer/inference/test_parameters.cpp
  holoinfer/inference/test_core.hpp
  holoinfer/inference/test_infer_settings.hpp
  holoinfer/processing/test_core.cpp
  holoinfer/processing/test_core.hpp
  holoinfer/processing/test_parameters.cpp
)

set_target_properties(HOLOINFER_TEST
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR}/gtests>"
    CXX_CLANG_TIDY ""
)

target_include_directories(HOLOINFER_TEST
  PRIVATE
    ${HOLOSCAN_TOP}/gxf_extensions
    ${CMAKE_SOURCE_DIR}/modules/holoinfer/src/include
)

target_link_libraries(HOLOINFER_TEST
  PRIVATE
    GTest::gmock_main
    GTest::gtest_main
    holoinfer
    holoscan::holoinfer_utils
    $<TARGET_NAME_IF_EXISTS:holoscan::infer::torch>
    yaml-cpp::yaml-cpp
)

set(_HOLOINFER_TEST_BIN_DIR ${${HOLOSCAN_PACKAGE_NAME}_BINARY_DIR})
gtest_discover_tests(HOLOINFER_TEST
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  DISCOVERY_MODE PRE_TEST
  PROPERTIES ENVIRONMENT
    "LD_LIBRARY_PATH=${_HOLOINFER_TEST_BIN_DIR}/lib:${_HOLOINFER_TEST_BIN_DIR}/gxf_extensions:${_HOLOINFER_TEST_BIN_DIR}/gxf_extensions/format_converter:${_HOLOINFER_TEST_BIN_DIR}/gxf_extensions/holoviz:${_HOLOINFER_TEST_BIN_DIR}/gxf_extensions/stream_playback:$ENV{LD_LIBRARY_PATH};HOLOSCAN_LIB_PATH=${_HOLOINFER_TEST_BIN_DIR}/gxf_extensions:${_HOLOINFER_TEST_BIN_DIR}/lib:$ENV{HOLOSCAN_LIB_PATH}"
)

install(
  TARGETS HOLOINFER_TEST
  COMPONENT holoscan-testing
  DESTINATION bin/gtests/libholoscan
  EXCLUDE_FROM_ALL
)

if(HOLOSCAN_BUILD_ORT)
  add_dependencies(HOLOINFER_TEST holoinfer_ort)
endif()

# ##################################################################################################
# * Flow Tracking tests ----------------------------------------------------------------------------------
ConfigureTest(
  FLOW_TRACKING_TEST
  flow_tracking/add_probe_operator_tests.cpp
  flow_tracking/async_buffer_tracking.cpp
  flow_tracking/async_buffer_tracking_cycles.cpp
  flow_tracking/entity_passthrough.cpp
  flow_tracking/flow_tracking_cycle.cpp
  flow_tracking/flow_tracking_graphs.cpp
  flow_tracking/limited_tracking_tests.cpp
  system/env_wrapper.cpp
)
target_link_libraries(FLOW_TRACKING_TEST
  PRIVATE
  holoscan::ops::ping_rx
  holoscan::ops::ping_tx
)
