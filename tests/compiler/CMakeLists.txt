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
        ${CMAKE_SOURCE_DIR}/src
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
    test_endpoint_catalog.cpp
    test_compile_lifecycle.cpp
    test_compile_lowering.cpp
    test_compile_parameters.cpp
    test_compile_partition_authoring.cpp
    physical_plan_test_support.cpp
    test_compile_physical_plan.cpp
    test_compile_physical_plan_artifact.cpp
    test_compile_physical_plan_tensor.cpp
    test_compile_physical_plan_validation.cpp
    test_compile_process_placement.cpp
    test_compile_temporal.cpp
    test_interprocess_partition_boundary.cpp
    test_execution_plan.cpp
    test_partition_environment.cpp
    test_partitioning.cpp
    test_profile_admission.cpp
    ../runtime/test_runtime_package_capability.cpp
)
holoscan_configure_test_clang_tidy(compile_test)
holoscan_apply_logging_policy(compile_test)

target_include_directories(compile_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(compile_test
    PRIVATE
        holoscan::core
        holoscan::schema
        interprocess_test_operators
        GTest::gtest_main
)

gtest_discover_tests(compile_test
    DISCOVERY_MODE PRE_TEST
    TEST_PREFIX COMPILE.
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Exercise the process environment capture contract in genuinely fresh
# processes. The probe deliberately does not mutate its own environment; the
# CMake driver launches one process for each accepted and rejected selector.
add_executable(partition_environment_process_probe
    $<TARGET_OBJECTS:holoscan_compiler_private_objects>
    partition_environment_process_probe.cpp
)
holoscan_configure_test_clang_tidy(partition_environment_process_probe)
holoscan_apply_logging_policy(partition_environment_process_probe)

target_include_directories(partition_environment_process_probe
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(partition_environment_process_probe
    PRIVATE
        holoscan::core
        interprocess_test_operators
)

# REQ: CHAN-085
add_test(NAME COMPILE.PartitionEnvironmentFreshProcessTest.ExactSelectorCases
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_PARTITION_ENVIRONMENT_PROBE=$<TARGET_FILE:partition_environment_process_probe>"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/run_partition_environment_fresh_process.cmake
)
set_tests_properties(COMPILE.PartitionEnvironmentFreshProcessTest.ExactSelectorCases
    PROPERTIES
        LABELS "ea1;compile;partitioning;environment"
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # REQ: CHAN-085
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.CoalescedFanoutCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-fanout
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.CoalescedFanoutCrossesExecBoundary
        PROPERTIES
            LABELS "ea1;compile;channel;interprocess;fanout"
            TIMEOUT 30
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectTensorLoanCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-tensor
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectTensorLoanCrossesExecBoundary
        PROPERTIES
            LABELS "ea1;compile;channel;interprocess;tensor"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectPinnedTensorLoanCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-pinned-tensor
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectPinnedTensorLoanCrossesExecBoundary
        PROPERTIES
            LABELS "ea1;compile;channel;interprocess;tensor;cuda"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundary
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-device-tensor
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundary
        PROPERTIES
            LABELS "ea1;compile;channel;interprocess;tensor;cuda"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundaryNonzeroDevice
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-device-tensor-nonzero
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectCudaTensorLoanCrossesExecBoundaryNonzeroDevice
        PROPERTIES
            LABELS "ea1;compile;channel;interprocess;tensor;cuda"
            SKIP_REGULAR_EXPRESSION "SKIP: independent-process CUDA Tensor test requires GPU ordinal 1"
    )

    # REQ: CHAN-093
    add_test(NAME COMPILE.InterprocessPartitionBoundaryTest.DirectHostTensorLoanCrossesExecBoundaryNonzeroAmbientDevice
        COMMAND $<TARGET_FILE:partition_environment_process_probe> worker-host-tensor-nonzero-ambient
    )
    set_tests_properties(COMPILE.InterprocessPartitionBoundaryTest.DirectHostTensorLoanCrossesExecBoundaryNonzeroAmbientDevice
        PROPERTIES
            LABELS "ea1;compile;channel;interprocess;tensor;cuda"
            SKIP_REGULAR_EXPRESSION "SKIP: independent-process CUDA Tensor test requires GPU ordinal 1"
    )
endif()

# Exercise EA1-015's final-plan interprocess lowering under both accepted
# covers. These are intentionally named in the cumulative Channel partition
# parity family even though the assertions are compiler-side plan evidence.
foreach(partition_algorithm IN ITEMS serial_baseline source_chain)
    set(interprocess_parity_test
        holoscan_channel_partition_parity_${partition_algorithm}_interprocess_locality)
    add_test(
        NAME ${interprocess_parity_test}
        COMMAND ${CMAKE_COMMAND} -E env
            "HOLOSCAN_EXPERIMENTAL_PARTITION_ALGORITHM=${partition_algorithm}/v1"
            $<TARGET_FILE:compile_test>
            --gtest_filter=InterprocessPartitionBoundaryTest.AcceptedCoverPreservesLocalityAndLogicalEdgeIdentity
    )
    set_tests_properties(${interprocess_parity_test}
        PROPERTIES
            LABELS "ea1;compile;channel;partition-parity;interprocess"
    )
endforeach()

add_test(NAME COMPILE_PURITY.no_forbidden_mutation_calls
    COMMAND ${CMAKE_COMMAND}
        "-DHOLOSCAN_SOURCE_DIR=${CMAKE_SOURCE_DIR}"
        -P ${CMAKE_CURRENT_SOURCE_DIR}/check_compile_purity.cmake
)
set_tests_properties(COMPILE_PURITY.no_forbidden_mutation_calls
    PROPERTIES
        LABELS "ea1;compile;purity;structural"
)
