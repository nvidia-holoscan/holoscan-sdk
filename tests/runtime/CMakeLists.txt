# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_library(runtime_test_graph_support OBJECT runtime_test_graph.cpp)
holoscan_configure_test_clang_tidy(runtime_test_graph_support)
holoscan_apply_logging_policy(runtime_test_graph_support)

target_include_directories(runtime_test_graph_support
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
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
    ${holoscan_SOURCE_DIR}/src/compiler/closed_island_activation_plan.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/compile_platform.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/execution_plan_lookup.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/inline_partition_execution_plan.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/partition_authority_validation.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/partition_runner_plan.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/process_local_channel_plan_validation.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/optional_input_plan_validation.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/profile_admission.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/runtime_package_capabilities.cpp
    ${holoscan_SOURCE_DIR}/src/core/diagnostic_context.cpp
    ${holoscan_SOURCE_DIR}/src/core/parameter_catalog_digest_internal.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/dispatcher.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/endpoint_clock_health_runtime.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/executor.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/lifecycle_control_state.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/lifecycle_executor.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/physical_readiness_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_notification_source.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/parameter_catalog_projection.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/parameter_control_state.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/parameter_update_coordinator.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_materialization.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_rate_state.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_activation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_binding_test_access.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_ingress.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_materialization.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_output.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_plan_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_restart.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_storage.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_clock_reconstruction.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_operator_reconstruction.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_operator_plan_validation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_partition_plan_validation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_operator_replay_validation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_package_capability.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_tensor_resources.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_tensor_output_pool.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_tensor_output_plan.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_tensor_resource_materialization.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/scheduling/deterministic_serial_ready_policy.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/scheduling/ready_policy_catalog.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/execution_lane.cpp
    ${holoscan_SOURCE_DIR}/src/threading/linux_thread_controller.cpp
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
    test_runtime_rate_state.cpp
    nvtx_emit_marker_capture.cpp
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
    test_runtime_partition_execution.cpp
    test_channel_lifecycle_restart.cpp
    test_runtime_profile.cpp
    test_reusable_graph_clock.cpp
    test_retained_tensor_readiness.cpp
    test_temporal_freshness.cpp
    test_runtime_inline_partition_execution.cpp
    test_runtime_diagnostic_sink.cpp
    run_session_channel_execution_test_support.cpp
    test_run_session.cpp
    test_optional_inputs.cpp
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
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(runtime_test
    PRIVATE
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        holomq::holomq
        runtime_test_graph_support
        GTest::gtest_main
        Threads::Threads
)

# Compile the private Channel mutation seams only into this source-tree test executable. The
# production Core translation unit sees only the friend forward declaration.
target_compile_definitions(runtime_test PRIVATE HOLOSCAN_RUNTIME_CHANNEL_TEST_ACCESS=1)

gtest_discover_tests(runtime_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX RUNTIME.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(runtime_signal_process_probe
        runtime_signal_process_probe.cpp
        ../support/direct_port_binding.cpp
    )
    holoscan_configure_test_clang_tidy(runtime_signal_process_probe)
    holoscan_apply_logging_policy(runtime_signal_process_probe)
    target_include_directories(runtime_signal_process_probe
        PRIVATE
            ${holoscan_SOURCE_DIR}/src
            ${holoscan_SOURCE_DIR}/tests
            ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(runtime_signal_process_probe
        PRIVATE
            holoscan::core
            runtime_test_graph_support
            Threads::Threads
    )

    set(managed_signal_scenarios
        complete
        sigint
        sigint-blocked
        sigterm
        domain-adopt
        domain-adopt-spent-retry
        async-managed
        domain-required
        caller-managed
        async-plain
        int-int
        term-term
        int-term
        term-int
        second-ignore
        second-custom
        persistent-second-default
        persistent-second-ignore
        persistent-second-custom
        persistent-second-custom-originally-blocked
        # REQ: RUNTIME-041
        # REQ: RUNTIME-049
        automatic-multi-owner-signals
        automatic-multi-owner-child-failure
        third-after-escalation-custom
        asymmetric-mask
        first-custom-preserved
        close-after-wait
        wrong-thread-close
        owner-busy
        caller-managed-domain
        wrong-thread-bind
        fork-rejected
        exec-mask-inheritance-and-remedies
        fork-locked-close-rejected
        fork-locked-destroy-rejected
        fork-locked-create-rejected
        fork-locked-default-run-rejected
        resource-loop
        owner-busy-establishing
        census-changed-during-setup
        pending-during-establishment
        pending-establishment-rollback
        failure-adopt-binding
        worker-start-failure
        close-while-bound
        unbound-destructor
        generation-monotonic
        generation-exhaustion
        invalid-policy
        failure-setup
        failure-start-broker
        failure-broker-task-open
        failure-setup-cleanup
        idle-bound-event-driven
        idle-unbound-event-driven
        terminal-wake-before-close
        failure-bound-wait-no-spin
        failure-unbound-wait-retires
        failure-notify
        notify-failure-blocked-controller
        failure-drain
        failure-fixed-state
        failure-wait-cleanup
        failure-persistent-drain
        failure-poll
        failure-read-signal
        failure-final-drain
        failure-terminal-release
        failure-close-signal
        failure-close-notification
        failure-restore-mask
        failure-close-domain
        failure-join-broker
        failure-broker-retirement
        failure-delegate-signal
        failure-second-restore-mask
        failure-persistent-second-restore-mask
        disposition-drift
        disposition-drift-transient
        signal-vs-failure
        signal-vs-requested-completion
        pending-before-attach
        broker-thread-directed
        same-signal-coalesced
        mixed-signals-pending
        signal-at-completed-candidate
        signal-at-failed-candidate
        signal-after-freeze-forwarded
        blocked-signal-after-freeze-pending
        liveness-cooperative
        liveness-polling
        liveness-uncooperative
    )
    foreach(scenario IN LISTS managed_signal_scenarios)
        add_test(
            NAME RUNTIME.RunSession.ManagedSignals.${scenario}
            COMMAND runtime_signal_process_probe ${scenario}
        )
        set_tests_properties(RUNTIME.RunSession.ManagedSignals.${scenario}
            PROPERTIES
                LABELS "ea2;runtime;run-session;signals"
                TIMEOUT 30
        )
    endforeach()
endif()

# REQ: CHAN-103
add_test(
    NAME RUNTIME.host_compute_cuda_fast_path
    COMMAND ${CMAKE_COMMAND}
        "-DRUN_SESSION_ACTIVATION_SOURCE=${holoscan_SOURCE_DIR}/src/runtime/run_session_activation.cpp"
        "-DGRAPH_EXEC_HEADER=${holoscan_SOURCE_DIR}/src/runtime/graph_exec.hpp"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/verify_host_compute_cuda_fast_path.cmake
)
set_tests_properties(
    RUNTIME.host_compute_cuda_fast_path
    PROPERTIES LABELS "runtime;channel;structural"
)

# Keep the two externally visible production-default parity gates while isolating
# HoloMQ-backed semantics in fresh processes so one teardown cannot contaminate the next.
set(parity_prefix holoscan_channel_partition_parity_source_chain)
set(trigger_test ${parity_prefix}_all_triggers_and_selectors)
set(cache_test ${parity_prefix}_cache_readiness_metadata_and_epochs)
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
        "-DSEMANTIC_FILTERS=${trigger_semantics_arg}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_channel_partition_parity.cmake
)
add_test(
    NAME ${cache_test}
    COMMAND ${CMAKE_COMMAND}
        "-DRUNTIME_TEST=$<TARGET_FILE:runtime_test>"
        "-DSEMANTIC_FILTERS=${cache_semantics_arg}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_channel_partition_parity.cmake
)
set_tests_properties(
    ${trigger_test}
    ${cache_test}
    PROPERTIES
        LABELS "runtime;channel;partition-parity"
)

add_executable(runtime_ready_policy_test
    ../common/allocation_counter.cpp
    test_ready_policy.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/scheduling/deterministic_serial_ready_policy.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/scheduling/ready_policy_catalog.cpp
)
holoscan_configure_test_clang_tidy(runtime_ready_policy_test)
holoscan_apply_logging_policy(runtime_ready_policy_test)

target_include_directories(runtime_ready_policy_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
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
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)
