# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_library(runtime_test_graph_support OBJECT runtime_test_graph.cpp)
holoscan_configure_test_clang_tidy(runtime_test_graph_support)
holoscan_apply_logging_policy(runtime_test_graph_support)

target_include_directories(runtime_test_graph_support
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(runtime_test_graph_support
    PUBLIC
        holoscan::core
        Threads::Threads
)

add_executable(runtime_test
    ../common/allocation_counter.cpp
    ../support/direct_port_binding.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/closed_island_activation_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/compile_platform.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/inline_partition_execution_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/partition_runner_plan.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/process_local_channel_plan_validation.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/profile_admission.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/runtime_package_capabilities.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/dispatcher.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/endpoint_clock_health_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/executor.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/lifecycle_control_state.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/lifecycle_executor.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/physical_readiness_binding.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_notification_source.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/parameter_catalog_projection.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/parameter_control_state.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/parameter_update_coordinator.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_materialization.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_channel_binding.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_package_capability.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduling/deterministic_serial_ready_policy.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduling/ready_policy_catalog.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/execution_lane.cpp
    ${CMAKE_SOURCE_DIR}/src/threading/linux_thread_controller.cpp
    test_deadline_runtime.cpp
    test_dispatcher.cpp
    test_dispatcher_binding.cpp
    test_endpoint_clock_health.cpp
    test_endpoint_clock_health_runtime.cpp
    test_lifecycle_control.cpp
    parameter_update_test_support.cpp
    test_parameter_updates.cpp
    test_parameter_updates_runtime.cpp
    test_physical_readiness_binding.cpp
    test_runtime_materialization.cpp
    test_runtime_notification_source.cpp
    runtime_channel_binding_test_support.cpp
    runtime_channel_binding_schema_tensor_test_support.cpp
    test_runtime_channel_binding.cpp
    test_runtime_channel_binding_schema_tensor.cpp
    test_runtime_channel_binding_schema_transport.cpp
    test_runtime_channel_binding_tensor_selection.cpp
    test_runtime_channel_binding_temporal.cpp
    test_runtime_channel_binding_lifecycle.cpp
    test_runtime_external_channel.cpp
    test_runtime_external_pod.cpp
    test_runtime_external_schema.cpp
    test_runtime_package_capability.cpp
    test_channel_lifecycle_restart.cpp
    test_runtime_profile.cpp
    test_reusable_graph_clock.cpp
    test_retained_tensor_readiness.cpp
    test_runtime_inline_partition_execution.cpp
    test_runtime_diagnostic_sink.cpp
    run_session_channel_execution_test_support.cpp
    test_run_session.cpp
    test_run_session_channel_execution.cpp
    test_run_session_descriptor_output.cpp
    test_run_session_lane_evidence.cpp
    test_run_session_plan_ownership.cpp
    test_run_session_tensor.cpp
    test_execution_lane.cpp
    $<TARGET_OBJECTS:holoscan_lifecycle_plan_private_objects>
)
holoscan_configure_test_clang_tidy(runtime_test)
holoscan_apply_logging_policy(runtime_test)

target_include_directories(runtime_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(runtime_test
    PRIVATE
        holoscan::core
        holoscan::schema
        runtime_test_graph_support
        GTest::gtest_main
        Threads::Threads
)

gtest_discover_tests(runtime_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX RUNTIME.
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# REQ: CHAN-103
add_test(
    NAME RUNTIME.host_compute_cuda_fast_path
    COMMAND ${CMAKE_COMMAND}
        "-DRUN_SESSION_SOURCE=${CMAKE_SOURCE_DIR}/src/runtime/run_session.cpp"
        "-DGRAPH_EXEC_HEADER=${CMAKE_SOURCE_DIR}/src/runtime/graph_exec.hpp"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_host_compute_cuda_fast_path.cmake
)
set_tests_properties(
    RUNTIME.host_compute_cuda_fast_path
    PROPERTIES LABELS "ea1;runtime;channel;structural"
)

# Keep four externally visible parity gates while isolating HoloMQ-backed
# semantics in fresh processes so one teardown cannot contaminate the next.
foreach(partition_algorithm IN ITEMS serial_baseline source_chain)
    set(parity_prefix
        holoscan_channel_partition_parity_${partition_algorithm})
    set(trigger_test ${parity_prefix}_all_triggers_and_selectors)
    set(cache_test ${parity_prefix}_cache_readiness_metadata_and_epochs)
    set(partition_selector
        HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM=${partition_algorithm}/v1)
    set(trigger_semantics
        ChannelPartitionPlanParityTest.AllTriggersAndSelectors
        ChannelPartitionPlanParityTest.InputAnyExecutesThroughChannelBinding
        ChannelPartitionPlanParityTest.InputEachExecutesThroughChannelBinding
        ChannelPartitionPlanParityTest.ScalarQueuePolicyDoesNotRedefineSelectionAndFreshRunResets
        RunSessionTest.ChannelInputPredicateMatchesDirectOracleAcrossTheBehaviorMatrix
        RunSessionTest.ChannelWindowSuccessorKeepsAnchorButUsesLaterActivationTime
        RunSessionTest.ChannelWindowTimeoutInvokesWithMissingWindowSelection
        ChannelLifecycleRestart.NextComputeLatestWithinUsesOnePinnedGenerationAcrossSelectorRoutes)
    set(cache_semantics
        ChannelPartitionPlanParityTest.CacheReadinessMetadataAndEpochs)
    list(JOIN trigger_semantics "|" trigger_semantics_arg)
    list(JOIN cache_semantics "|" cache_semantics_arg)
    add_test(
        NAME ${trigger_test}
        COMMAND ${CMAKE_COMMAND}
            "-DRUNTIME_TEST=$<TARGET_FILE:runtime_test>"
            "-DPARTITION_SELECTOR=${partition_selector}"
            "-DSEMANTIC_FILTERS=${trigger_semantics_arg}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_channel_partition_parity.cmake
    )
    add_test(
        NAME ${cache_test}
        COMMAND ${CMAKE_COMMAND}
            "-DRUNTIME_TEST=$<TARGET_FILE:runtime_test>"
            "-DPARTITION_SELECTOR=${partition_selector}"
            "-DSEMANTIC_FILTERS=${cache_semantics_arg}"
            -P ${CMAKE_CURRENT_SOURCE_DIR}/run_channel_partition_parity.cmake
    )
    set_tests_properties(
        ${trigger_test}
        ${cache_test}
        PROPERTIES
            LABELS "ea1;runtime;channel;partition-parity"
    )
endforeach()

add_executable(runtime_ready_policy_test
    ../common/allocation_counter.cpp
    test_ready_policy.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduling/deterministic_serial_ready_policy.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduling/ready_policy_catalog.cpp
)
holoscan_configure_test_clang_tidy(runtime_ready_policy_test)
holoscan_apply_logging_policy(runtime_ready_policy_test)

target_include_directories(runtime_ready_policy_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(runtime_ready_policy_test
    PRIVATE
        holoscan::core
        GTest::gtest_main
)

gtest_discover_tests(runtime_ready_policy_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX holoscan_runtime_ready_policy_
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
