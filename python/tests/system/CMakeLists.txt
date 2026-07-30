# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Note: test_pytracing.py is excluded — it uses a non-pytest invocation and is
# handled separately in python/CMakeLists.txt.
set(PYTHON_SYSTEM_TESTS
  test_add_arg_after_operator_init.py
  test_application_async_ping.py
  test_application_broadcast.py
  test_application_exception.py
  test_application_invalid_native_op_params.py
  test_application_minimal.py
  test_application_multi_receiver_ping.py
  test_application_ping_core_pinning.py
  test_application_ping_none_condition.py
  test_application_ping.py
  test_application_realtime_ping.py
  test_application_video_stream.py
  test_application_with_invalid_add_flow.py
  test_application_with_repeated_emit_on_same_port.py
  test_cpp_types_to_from_python.py
  test_cuda_stream_condition.py
  test_decorator_apps.py
  test_flow_tracking_with_tensormap.py
  test_format_converter_invalid_input.py
  test_holoviz_camera_emit_receive.py
  test_holoviz_dual_window.py
  test_holoviz_window_close_callback.py
  test_memory_available_condition.py
  test_multiple_run.py
  test_multithread_tensor_message.py
  test_native_conditions.py
  test_operator_initialization_order.py
  test_operator_metadata_and_data_logging.py
  test_operator_receiver_and_transmitter_methods.py
  test_operator_status.py
  test_operator_tensor_validation.py
  test_or_combiner.py
  test_port_map_description.py
  test_pose_tree_manager_service_lookup.py
  test_positional_condition_args.py
  test_pytorch_sanity_check.py
  test_queue_size_warning.py
  test_receivers_pyentity_refcount.py
  test_receivers_pyobject_refcount.py
  test_rx_tx_queue_policy.py
  test_stream_handling.py
  test_stress_testing.py
  test_subgraph.py
  test_timestamp_emit_receive.py
)

holoscan_add_pytest_tests(
  FILES ${PYTHON_SYSTEM_TESTS}
  TEST_DIRECTORY "${CMAKE_PYTHON_WORKING_DIR}/tests/system"
  WORKING_DIRECTORY ${CMAKE_PYTHON_WORKING_DIR}
  PREFIX python-api-system
  PYTEST_ARGS -v --durations=0 --run-realtime
  TIMEOUT 1000
)
