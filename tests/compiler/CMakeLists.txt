# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(GoogleTest)

add_library(interprocess_test_operators STATIC
    interprocess_test_operators.cpp
)
holoscan_configure_test_clang_tidy(interprocess_test_operators)
holoscan_apply_logging_policy(interprocess_test_operators)

target_include_directories(interprocess_test_operators
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(interprocess_test_operators
    PRIVATE
        holoscan::core
)

# The private compiler substrate deliberately has no exported or installed ABI.
# Compile the cold-path lowering and planner sources into the conformance target
# so adversarial tests exercise the exact source-tree seams without making those
# seams public.
add_executable(compile_test
    $<TARGET_OBJECTS:holoscan_compiler_private_objects>
    $<TARGET_OBJECTS:holoscan_lifecycle_plan_private_objects>
    test_compile.cpp
    test_compile_completeness.cpp
    test_compile_deployment.cpp
    test_compile_green_context_authoring.cpp
    test_endpoint_catalog.cpp
    test_compile_lifecycle.cpp
    test_compile_lowering.cpp
    test_compile_parameters.cpp
    test_compile_partition_authoring.cpp
    artifact_test_support.cpp
    physical_plan_test_support.cpp
    test_compile_physical_plan.cpp
    test_compile_physical_plan_artifact.cpp
    test_compile_physical_plan_tensor.cpp
    test_compile_physical_plan_validation.cpp
    test_compile_process_placement.cpp
    test_compile_rate.cpp
    test_compile_temporal.cpp
    test_compile_tensor_byte_span_derivation.cpp
    test_execution_plan_contracts.cpp
    test_execution_plan_identity.cpp
    test_execution_plan_lookup.cpp
    test_plan_lowering_diagnostic.cpp
    test_plan_record_contracts.cpp
    test_interprocess_partition_boundary.cpp
    test_execution_plan.cpp
    test_partition_environment.cpp
    test_partitioning.cpp
    test_profile_admission.cpp
    ../runtime/test_runtime_package_capability.cpp
    ../common/allocation_counter.cpp
)
holoscan_configure_test_clang_tidy(compile_test)
holoscan_apply_logging_policy(compile_test)

target_include_directories(compile_test
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

set(compiler_artifact_golden_dir
    "${CMAKE_CURRENT_BINARY_DIR}/testdata")
file(MAKE_DIRECTORY "${compiler_artifact_golden_dir}")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/testdata/zero_binding_tensor_graph.v4.normalized.json"
    "${compiler_artifact_golden_dir}/zero_binding_tensor_graph.v4.normalized.json"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/testdata/green_context_partitions.v4.normalized.json"
    "${compiler_artifact_golden_dir}/green_context_partitions.v4.normalized.json"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/testdata/lifecycle_on_each_inverse_rate_horizons.v4.normalized.json"
    "${compiler_artifact_golden_dir}/lifecycle_on_each_inverse_rate_horizons.v4.normalized.json"
    COPYONLY
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/testdata/tensor_byte_span_derivation.v2.json"
    "${compiler_artifact_golden_dir}/tensor_byte_span_derivation.v2.json"
    COPYONLY
)
target_compile_definitions(compile_test
    PRIVATE
        HOLOSCAN_COMPILER_TEST_DATA_DIR="${compiler_artifact_golden_dir}"
)

target_link_libraries(compile_test
    PRIVATE
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        interprocess_test_operators
        GTest::gtest_main
)

gtest_discover_tests(compile_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX COMPILE.
    WORKING_DIRECTORY ${holoscan_BINARY_DIR}
)

# Exercise the removed-selector contract in genuinely fresh processes. The
# probe deliberately does not mutate its own environment; the CMake driver
# launches one default case and multiple value-independent rejection cases.
add_executable(partition_environment_process_probe
    $<TARGET_OBJECTS:holoscan_compiler_private_objects>
    partition_environment_process_probe.cpp
)
holoscan_configure_test_clang_tidy(partition_environment_process_probe)
holoscan_apply_logging_policy(partition_environment_process_probe)

target_include_directories(partition_environment_process_probe
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(partition_environment_process_probe
    PRIVATE
        holoscan::core
        interprocess_test_operators
)

# REQ: COMPILE-068
add_test(NAME COMPILE.PartitionEnvironmentFreshProcessTest.ExactSelectorCases
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_PARTITION_ENVIRONMENT_PROBE=$<TARGET_FILE:partition_environment_process_probe>"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_partition_environment_fresh_process.cmake
)
set_tests_properties(COMPILE.PartitionEnvironmentFreshProcessTest.ExactSelectorCases
    PROPERTIES
        LABELS "ea2;compile;partitioning;environment"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # REQ: COMPILE-068
    add_test(NAME COMPILE.PartitionEnvironmentFreshProcessTest.WorkerRuntimePoison
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-default
    )
    set_tests_properties(COMPILE.PartitionEnvironmentFreshProcessTest.WorkerRuntimePoison
        PROPERTIES
            LABELS "ea2;compile;partitioning;environment;interprocess"
            TIMEOUT 30
    )

    # REQ: CHAN-085
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.CoalescedFanoutCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-fanout
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.CoalescedFanoutCrossesExecBoundary
        PROPERTIES
            LABELS "compile;channel;interprocess;fanout"
            TIMEOUT 30
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectTensorLoanCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-tensor
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectTensorLoanCrossesExecBoundary
        PROPERTIES
            LABELS "compile;channel;interprocess;tensor"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectPinnedTensorLoanCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-pinned-tensor
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectPinnedTensorLoanCrossesExecBoundary
        PROPERTIES
            LABELS "compile;channel;interprocess;tensor;cuda"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-device-tensor
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundary
        PROPERTIES
            LABELS "compile;channel;interprocess;tensor;cuda"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundaryNonzeroDevice
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-device-tensor-nonzero
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundaryNonzeroDevice
        PROPERTIES
            LABELS "compile;channel;interprocess;tensor;cuda"
            SKIP_REGULAR_EXPRESSION "SKIP: independent-process CUDA Tensor test requires GPU ordinal 1"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectHostTensorLoanCrossesExecBoundaryNonzeroAmbientDevice
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-host-tensor-nonzero-ambient
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectHostTensorLoanCrossesExecBoundaryNonzeroAmbientDevice
        PROPERTIES
            LABELS "compile;channel;interprocess;tensor;cuda"
            SKIP_REGULAR_EXPRESSION "SKIP: independent-process CUDA Tensor test requires GPU ordinal 1"
    )
endif()

# Exercise HSDK5-015's final-plan interprocess lowering under the production
# source-chain cover. The test remains compiler-side plan evidence.
add_test(
    NAME holoscan_channel_partition_parity_source_chain_interprocess_locality
    COMMAND $<TARGET_FILE:compile_test>
        --gtest_filter=InterprocessPartitionBoundaryTest.AcceptedCoverPreservesLocalityAndLogicalEdgeIdentity
)
set_tests_properties(
    holoscan_channel_partition_parity_source_chain_interprocess_locality
    PROPERTIES
        LABELS "ea2;compile;channel;partition-parity;interprocess"
)

add_test(NAME COMPILE_PURITY.no_forbidden_mutation_calls
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SOURCE_DIR=${holoscan_SOURCE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_compile_purity.cmake
)
set_tests_properties(COMPILE_PURITY.no_forbidden_mutation_calls
    PROPERTIES
        LABELS "compile;purity;structural"
)
