# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(runtime_dispatcher_benchmark
    runtime_dispatcher_benchmark.cpp
    ${holoscan_SOURCE_DIR}/tests/support/direct_port_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/dispatcher.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/physical_readiness_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/scheduling/deterministic_serial_ready_policy.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/scheduling/ready_policy_catalog.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/execution_lane.cpp
    ${holoscan_SOURCE_DIR}/src/threading/linux_thread_controller.cpp
)
add_dependencies(holoscan-benchmark-targets runtime_dispatcher_benchmark)
target_include_directories(runtime_dispatcher_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
)
target_link_libraries(runtime_dispatcher_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

add_executable(runtime_channel_benchmark
    runtime_channel_benchmark.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/execution_plan_lookup.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/process_local_channel_plan_validation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/endpoint_clock_health_runtime.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_activation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_binding_test_access.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_ingress.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_materialization.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_output.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_plan_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_restart.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_storage.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/execution_lane.cpp
    ${holoscan_SOURCE_DIR}/src/threading/linux_thread_controller.cpp
)
add_dependencies(holoscan-benchmark-targets runtime_channel_benchmark)
target_include_directories(runtime_channel_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
)
target_compile_definitions(runtime_channel_benchmark
    PRIVATE
        HOLOSCAN_RUNTIME_CHANNEL_TEST_ACCESS=1
)
target_link_libraries(runtime_channel_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
        Threads::Threads
)

add_executable(runtime_payload_benchmark
    runtime_payload_benchmark.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/execution_plan_lookup.cpp
    ${holoscan_SOURCE_DIR}/src/compiler/process_local_channel_plan_validation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/endpoint_clock_health_runtime.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_activation.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_ingress.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_materialization.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_output.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_plan_binding.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_restart.cpp
    ${holoscan_SOURCE_DIR}/src/runtime/runtime_channel_storage.cpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_payload_benchmark holoscan_flatbuffer_schemas_generated)
add_dependencies(holoscan-benchmark-targets runtime_payload_benchmark)
target_include_directories(runtime_payload_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
)
target_link_libraries(runtime_payload_benchmark
    PRIVATE
        benchmark::benchmark_main
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        Threads::Threads
)

set(RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/runtime_multi_tensor_generated")
holoscan_add_flatbuffer_schema(
    TARGET runtime_multi_tensor_benchmark_generated
    SCHEMAS runtime_multi_tensor_benchmark.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
    BFBS_EMBED
    DESCRIPTOR_TENSOR
    TRAITS_HEADER runtime_multi_tensor_benchmark_traits.hpp
    TRAITS_ROOT_TYPE runtime_benchmark::MultiTensorMessage
    TRAITS_NATIVE_TYPE runtime_benchmark::MultiTensorMessageT
    TRAITS_BINARY_SCHEMA_TYPE runtime_benchmark::MultiTensorMessageBinarySchema
    TRAITS_PACKAGE_NAMESPACE runtime_benchmark::package
    TRAITS_CANONICAL_NAME runtime_benchmark.MultiTensorMessage
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 1536
    TRAITS_DESCRIPTOR_METADATA_CAPACITY 512
)

set(RUNTIME_THIRD_PARTY_BENCHMARK_GENERATED_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/runtime_third_party_generated")
holoscan_add_flatbuffer_schema(
    TARGET runtime_third_party_benchmark_generated
    SCHEMAS runtime_third_party_benchmark.fbs
    INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_DIR "${RUNTIME_THIRD_PARTY_BENCHMARK_GENERATED_DIR}"
    BFBS_EMBED
    TRAITS_HEADER runtime_third_party_benchmark_traits.hpp
    TRAITS_ROOT_TYPE task30_benchmark::Telemetry
    TRAITS_NATIVE_TYPE task30_benchmark::TelemetryT
    TRAITS_BINARY_SCHEMA_TYPE task30_benchmark::TelemetryBinarySchema
    TRAITS_PACKAGE_NAMESPACE task30_benchmark::package
    TRAITS_CANONICAL_NAME task30_benchmark.Telemetry
    TRAITS_SCHEMA_VERSION 1
    TRAITS_MAX_SERIALIZED_SIZE 128
)

add_executable(runtime_graph_benchmark
    runtime_graph_benchmark.cpp
    runtime_graph_benchmark_common.hpp
    ../common/transport_benchmark_common.hpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_graph_benchmark
    holoscan_flatbuffer_schemas_generated
    runtime_multi_tensor_benchmark_generated
    runtime_third_party_benchmark_generated
)
add_dependencies(holoscan-benchmark-targets runtime_graph_benchmark)
target_include_directories(runtime_graph_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
        "${RUNTIME_THIRD_PARTY_BENCHMARK_GENERATED_DIR}"
)
target_link_libraries(runtime_graph_benchmark
    PRIVATE
        flatbuffers::flatbuffers
        fmt::fmt-header-only
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        Threads::Threads
)

add_executable(runtime_graph_interprocess_benchmark
    runtime_graph_interprocess_benchmark.cpp
    runtime_graph_benchmark_common.hpp
    ../common/transport_benchmark_common.hpp
    ${holoscan_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_graph_interprocess_benchmark
    holoscan_flatbuffer_schemas_generated
    runtime_multi_tensor_benchmark_generated
)
add_dependencies(holoscan-benchmark-targets runtime_graph_interprocess_benchmark)
target_include_directories(runtime_graph_interprocess_benchmark
    PRIVATE
        ${holoscan_SOURCE_DIR}/src
        ${holoscan_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
)
target_link_libraries(runtime_graph_interprocess_benchmark
    PRIVATE
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        holoscan::sensor_io
        Threads::Threads
)

if(BUILD_TESTING)
    foreach(green_context_layout IN ITEMS
            same_partition dedicated_to_dedicated dedicated_to_residual)
        if(green_context_layout STREQUAL "same_partition")
            set(green_context_operator_lanes shared)
        else()
            set(green_context_operator_lanes separate)
        endif()
        string(TOUPPER "${green_context_layout}" green_context_layout_upper)
        add_test(
            NAME "BENCHMARK_GREEN_CONTEXT_${green_context_layout_upper}_SMOKE"
            COMMAND $<TARGET_FILE:runtime_graph_benchmark>
                    --scenario tensor
                    --tensor-storage device
                    --tensor-bytes 4096
                    --operator-lanes "${green_context_operator_lanes}"
                    --green-context-layout "${green_context_layout}"
                    --workload latency
                    --samples 8
                    --warmup 2
                    --queue-depth 1
                    --publisher-period-us 1000
                    --payload-validation all
                    --cuda-resource-counters
                    --timeout-ms 10000
                    --output
                    "${CMAKE_CURRENT_BINARY_DIR}/green_context_${green_context_layout}.json"
                    --plan-output
                    "${CMAKE_CURRENT_BINARY_DIR}/green_context_${green_context_layout}_plan.json"
        )
        set_tests_properties(
            "BENCHMARK_GREEN_CONTEXT_${green_context_layout_upper}_SMOKE"
            PROPERTIES
                LABELS "benchmark;ea2;runtime;cuda;green_context"
                SKIP_RETURN_CODE 77
                TIMEOUT 30
        )
    endforeach()

    if(Python3_Interpreter_FOUND)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_PYTHON_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_graph_benchmark.py"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER_PYTHON_TESTS
            PROPERTIES
                LABELS "benchmark;runtime"
        )

        add_test(
            NAME BENCHMARK_PARTITION_COVER_COMPARISON_PYTHON_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_run_partition_cover_comparison.py"
        )
        set_tests_properties(BENCHMARK_PARTITION_COVER_COMPARISON_PYTHON_TESTS
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
        )

        foreach(PARTITION_COVER IN ITEMS automatic include_all hybrid)
            set(PARTITION_COVER_LANES separate)
            if(PARTITION_COVER STREQUAL "include_all")
                set(PARTITION_COVER_LANES shared)
            endif()
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_${PARTITION_COVER}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --partition-cover "${PARTITION_COVER}"
                        --operator-lanes "${PARTITION_COVER_LANES}"
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 15
                        --repetitions 1
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_partition_cover_${PARTITION_COVER}"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_${PARTITION_COVER}
                PROPERTIES
                    LABELS "benchmark;ea2;runtime;partitioning"
                    TIMEOUT 30
            )
        endforeach()

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_SHARED_REJECTED
            COMMAND ${CMAKE_COMMAND}
                    "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                    -DREJECTION_CASE=automatic_shared
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/assert_runtime_graph_partition_rejected.cmake"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_SHARED_REJECTED
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
                TIMEOUT 15
        )

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FORCE_HOLOMQ_REJECTED
            COMMAND ${CMAKE_COMMAND}
                    "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                    -DREJECTION_CASE=automatic_forced_holomq
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/assert_runtime_graph_partition_rejected.cmake"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FORCE_HOLOMQ_REJECTED
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
                TIMEOUT 15
        )

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FANOUT_PHASE_TRACE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --partition-cover automatic
                    --operator-lanes separate
                    --workload notification_latency
                    --scenario pod
                    --fanout 16
                    --samples 256
                    --warmup 32
                    --publisher-period-us 1000
                    --payload-validation all
                    --runtime-phase-counters
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 3
                    --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_partition_cover_automatic_fanout_phase_trace"
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_PARTITION_COVER_AUTOMATIC_FANOUT_PHASE_TRACE
            PROPERTIES
                LABELS "benchmark;ea2;runtime;partitioning"
                TIMEOUT 60
        )

        if(Python3_VERSION VERSION_GREATER_EQUAL 3.10)
            add_test(
                NAME BENCHMARK_GREEN_CONTEXT_QUALIFICATION_PYTHON_TESTS
                COMMAND ${Python3_EXECUTABLE} -m pytest -q
                        "${CMAKE_CURRENT_SOURCE_DIR}/test_run_green_context_qualification.py"
            )
            set_tests_properties(BENCHMARK_GREEN_CONTEXT_QUALIFICATION_PYTHON_TESTS
                PROPERTIES
                    LABELS "benchmark;ea2;runtime;cuda"
            )
        endif()

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 2
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_smoke"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER
            PROPERTIES
                LABELS "benchmark;runtime"
                TIMEOUT 45
        )

        # Exercise the depth-one staged-ingress attribution through the Python runner so CI
        # validates the emitted counters, replacement evidence, and final delivery drain rather
        # than merely checking that the native benchmark exits successfully. This diagnostic arm
        # is intentionally distinct from counters-off performance baselines.
        add_test(
            NAME
                BENCHMARK_RUNTIME_GRAPH_RUNNER_DROP_OLDEST_DEPTH_ONE_HOLOMQ_FALLBACK_DIAGNOSTIC
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --operator-lanes separate
                    --queue-policy drop_oldest
                    --queue-depth 1
                    --notification-capacity 1
                    --samples 16
                    --warmup 4
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --test-gate-pod-sink-for-pressure
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 1
                    --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_drop_oldest_depth_one_holomq_fallback_diagnostic"
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_RUNNER_DROP_OLDEST_DEPTH_ONE_HOLOMQ_FALLBACK_DIAGNOSTIC
            PROPERTIES
                LABELS "benchmark;runtime;lossy;diagnostic"
                TIMEOUT 30
        )

        foreach(THIRD_PARTY_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(THIRD_PARTY_DELIVERY_ARGUMENTS)
            if(THIRD_PARTY_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND THIRD_PARTY_DELIVERY_ARGUMENTS
                    --force-holomq-process-local-inline)
            endif()
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_THIRD_PARTY_${THIRD_PARTY_DELIVERY}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --scenario third_party_schema
                        --operator-lanes shared
                        --callback-work-us 5
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --payload-validation all
                        --runtime-phase-counters
                        ${THIRD_PARTY_DELIVERY_ARGUMENTS}
                        --session-timeout-ms 10000
                        --process-timeout-sec 15
                        --repetitions 1
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_third_party_${THIRD_PARTY_DELIVERY}"
            )
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_RUNNER_THIRD_PARTY_${THIRD_PARTY_DELIVERY}
                PROPERTIES
                    LABELS "benchmark;runtime;schema;third-party"
                    TIMEOUT 30
            )
        endforeach()

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_RUNNER_PYTHON_TESTS
                COMMAND ${Python3_EXECUTABLE} -m pytest -q
                        "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_graph_interprocess_benchmark.py"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_RUNNER_PYTHON_TESTS
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
            )

            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TRACE_CORRELATION
                COMMAND $<TARGET_FILE:runtime_graph_interprocess_benchmark>
                        --run-id trace-correlation-self-test
                        --samples 1
                        --warmup 0
                        --trace-correlation-self-test 1
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TRACE_CORRELATION
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
            )

            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_START_BARRIER
                COMMAND $<TARGET_FILE:runtime_graph_interprocess_benchmark>
                        --run-id start-barrier-self-test
                        --samples 1
                        --warmup 0
                        --start-barrier-self-test 1
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_START_BARRIER
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
            )

            # Prove two independently compiled process-group workers exchange exact POD samples.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 1000
                        --payload-validation all
                        --runtime-phase-trace
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to1_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Exercise compiler-owned cross-process Tensor pools for each supported storage kind.
            foreach(tensor_storage IN ITEMS pageable_host pinned_host device)
                string(TOUPPER "${tensor_storage}" tensor_storage_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload tensor
                            --tensor-storage "${tensor_storage}"
                            --tensor-bytes 1048576
                            --workload latency
                            --samples 4
                            --warmup 1
                            --queue-depth 1
                            --publisher-period-us 1000
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_tensor_${tensor_storage}_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                # Keep transport-only throughput distinct from the full per-sample producer write.
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload tensor
                            --tensor-storage "${tensor_storage}"
                            --tensor-bytes 1048576
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 8
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 0
                            --payload-validation final
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_tensor_${tensor_storage}_descriptor_only_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_${tensor_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                if(tensor_storage STREQUAL "pageable_host")
                    add_test(
                        NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_PAGEABLE_HOST_DESCRIPTOR_ONLY_LATENCY_SMOKE
                        COMMAND ${Python3_EXECUTABLE}
                                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                                "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                                --payload tensor
                                --tensor-storage pageable_host
                                --tensor-bytes 1048576
                                --bulk-work-profile descriptor_only
                                --workload latency
                                --samples 16
                                --warmup 8
                                --queue-depth 1
                                --publisher-period-us 1000
                                --payload-validation final
                                --session-timeout-ms 10000
                                --process-timeout-sec 30
                                --output-dir
                                "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_tensor_pageable_host_descriptor_only_latency_smoke"
                    )
                    set_tests_properties(
                        BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_TENSOR_PAGEABLE_HOST_DESCRIPTOR_ONLY_LATENCY_SMOKE
                        PROPERTIES
                            LABELS "benchmark;runtime;interprocess;tensor;storage"
                            SKIP_RETURN_CODE 77
                            TIMEOUT 120
                    )
                endif()
            endforeach()

            # Keep Image metadata-plus-Tensor costs separate from raw Tensor transport, record the
            # intentional owning-receive copy allocations, and prove that every supported schema
            # provider preserves its declared memory kind.
            foreach(image_storage IN ITEMS pageable_host pinned_host device)
                string(TOUPPER "${image_storage}" image_storage_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_LATENCY_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload image
                            --tensor-storage "${image_storage}"
                            --tensor-bytes 1048576
                            --bulk-work-profile full_write
                            --workload latency
                            --samples 4
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 1000
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_image_${image_storage}_latency_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_LATENCY_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;schema"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload image
                            --tensor-storage "${image_storage}"
                            --tensor-bytes 1048576
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 8
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 0
                            --payload-validation final
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_image_${image_storage}_descriptor_only_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;schema"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )

                if(image_storage STREQUAL "pageable_host" OR image_storage STREQUAL "device")
                    add_test(
                        NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_LATENCY_SMOKE"
                        COMMAND ${Python3_EXECUTABLE}
                                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                                "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                                --payload image
                                --tensor-storage "${image_storage}"
                                --tensor-bytes 1048576
                                --bulk-work-profile descriptor_only
                                --workload latency
                                --samples 8
                                --warmup 8
                                --queue-depth 1
                                --publisher-period-us 1000
                                --payload-validation final
                                --session-timeout-ms 10000
                                --process-timeout-sec 30
                                --output-dir
                                "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_image_${image_storage}_descriptor_only_latency_smoke"
                    )
                    set_tests_properties(
                        "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_IMAGE_${image_storage_upper}_DESCRIPTOR_ONLY_LATENCY_SMOKE"
                        PROPERTIES
                            LABELS "benchmark;runtime;interprocess;tensor;schema;storage"
                            SKIP_RETURN_CODE 77
                            TIMEOUT 120
                    )
                endif()
            endforeach()

            # Exercise both multi-Tensor authoring paths through real Operator callbacks. The
            # excluded final warmup sample proves the direct bundle and staged-value callbacks are
            # allocation-free when the sink uses the activation-scoped borrowed batch view.
            foreach(multi_tensor_payload IN ITEMS multi_tensor_loan multi_tensor_staging)
                string(TOUPPER "${multi_tensor_payload}" multi_tensor_payload_upper)
                foreach(multi_tensor_storage IN ITEMS pageable_host pinned_host device)
                    string(TOUPPER "${multi_tensor_storage}" multi_tensor_storage_upper)
                    set(multi_tensor_storage_test
                        "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_${multi_tensor_storage_upper}_SMOKE")
                    add_test(
                            NAME "${multi_tensor_storage_test}"
                            COMMAND ${Python3_EXECUTABLE}
                                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                                    "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                                    --payload "${multi_tensor_payload}"
                                    --tensor-storage "${multi_tensor_storage}"
                                    --tensor-bytes 4096
                                    --bulk-work-profile descriptor_only
                                    --workload latency
                                    --samples 4
                                    --warmup 8
                                    --queue-depth 1
                                    --publisher-period-us 1000
                                    --payload-validation final
                                    --session-timeout-ms 10000
                                    --process-timeout-sec 30
                                    --output-dir
                                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_${multi_tensor_payload}_${multi_tensor_storage}_smoke"
                    )
                    set_tests_properties(
                        "${multi_tensor_storage_test}"
                        PROPERTIES
                            LABELS "benchmark;runtime;interprocess;tensor;schema;multi_tensor;storage"
                            SKIP_RETURN_CODE 77
                            TIMEOUT 120
                    )
                endforeach()
                # Saturation exhausts the multi-field reservation metadata before the next
                # envelope loan; keep this case to prevent reintroducing that progress cycle.
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_SATURATED_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload "${multi_tensor_payload}"
                            --tensor-storage pageable_host
                            --tensor-bytes 1048576
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 16
                            --warmup 16
                            --queue-depth 2
                            --publisher-period-us 0
                            --payload-validation final
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_${multi_tensor_payload}_saturated_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_SATURATED_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;tensor;schema;multi_tensor;saturated"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
            endforeach()

            # Exercise owning generated-schema encode/materialize across one process boundary and
            # require its fixed-field emit/receive callbacks to remain allocation-free.
            foreach(schema_workload IN ITEMS latency saturated)
                if(schema_workload STREQUAL "latency")
                    set(schema_publisher_period_us 1000)
                else()
                    set(schema_publisher_period_us 0)
                endif()
                string(TOUPPER "${schema_workload}" schema_workload_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_SCHEMA_TIME_${schema_workload_upper}_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload schema
                            --workload "${schema_workload}"
                            --samples 8
                            --warmup 2
                            --queue-depth 1
                            --publisher-period-us "${schema_publisher_period_us}"
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_schema_time_${schema_workload}_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_SCHEMA_TIME_${schema_workload_upper}_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;runtime;interprocess;schema"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
            endforeach()

            # Prove one source independently reaches two process-isolated consumers without loss.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --fanout 2
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 5000
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to2_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess;fanout"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Prove sustained coalesced fanout reports one logical publication and two exact,
            # ordered deliveries without partial visibility or duplicate retry.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SATURATED_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --fanout 2
                        --workload saturated
                        --samples 32
                        --warmup 8
                        --queue-depth 2
                        --publisher-period-us 0
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to2_saturated_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO2_SATURATED_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess;fanout"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Prove one exact stream crosses source, relay, and sink process groups in order.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_CHAIN_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --topology chain
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 1000
                        --payload-validation all
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_chain_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_CHAIN_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess;chain"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )

            # Exercise the low-latency ceiling with both Operator lanes continuously active.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_BUSY_POLL_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                        --workload latency
                        --samples 8
                        --warmup 2
                        --queue-depth 1
                        --publisher-period-us 1000
                        --payload-validation all
                        --busy-poll
                        --runtime-phase-trace
                        --session-timeout-ms 10000
                        --process-timeout-sec 30
                        --output-dir
                        "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_1to1_busy_poll_smoke"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_1TO1_BUSY_POLL_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;interprocess"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 120
            )
        endif()

        # Run the complete native/runner evidence path with one shared Operator lane.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_SHARED_LANE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload notification_latency
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 2
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_shared_lane_smoke"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER_SHARED_LANE
            PROPERTIES
                LABELS "benchmark;runtime"
                TIMEOUT 45
        )

        # Validate persisted role/lane and completion evidence for independently placed sinks.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_PER_SINK_LANES
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 4
                    --operator-lanes per_sink
                    --samples 8
                    --warmup 2
                    --queue-depth 2
                    --notification-capacity 2
                    --payload-validation all
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 2
                    --output-dir "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_runner_per_sink_lanes_smoke"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER_PER_SINK_LANES
            PROPERTIES
                LABELS "benchmark;runtime;fanout;placement"
                TIMEOUT 45
        )
    endif()

    string(CONCAT RUNTIME_CHANNEL_SMOKE_FILTER
        "BM_ChannelActivationCycleByGraphOwners/2"
        "|BM_ChannelReadinessBurst/1/1"
        "|BM_ChannelReadinessPollViaBusyLane/real_time"
        "|BM_ChannelDirectRetainedReadinessPoll/real_time"
        "|BM_ChannelDirectRetainedReadinessPollViaBusyLane/real_time"
        "|BM_ChannelDirectRetainedFusedPollConsumeCeiling/real_time"
        "|BM_ChannelDirectRetainedTwoDispatchPollConsume/real_time"
        "|BM_ChannelDirectRetainedResumablePollConsumeProbe/real_time"
        "|BM_ChannelDirectRetainedActivationCeiling/ordinary_sleeping/0/1/manual_time"
        "|BM_ChannelDirectRetainedActivationCeiling/receipt_sleeping/0/1/manual_time"
        "|BM_ChannelDirectRetainedActivationCeiling/composite_sleeping/0/1/manual_time"
        "|BM_ChannelLastWitnessRetirementByGraphInputs/1"
        "|BM_ChannelTypedPODCycle/real_time"
        "|BM_ChannelTypedPODDirectRetainedCycle/real_time"
        "|BM_ChannelTypedPOD64DirectRetainedCycle/real_time"
        "|BM_ChannelTypedPODPublishCut/manual_time"
        "|BM_ChannelTypedPODPublishRetire/manual_time"
        "|BM_ChannelTypedPODEmit/manual_time"
        "|BM_ChannelTypedPODDirectRetainedEmit/manual_time"
        "|BM_ChannelTypedPODFanoutEmit/4/manual_time"
        "|BM_ChannelTypedPODFanoutPublishCut/4/manual_time"
        "|BM_ChannelTypedPODFanoutConsumeRetire/4/manual_time"
        "|BM_ChannelTypedPODFanoutFusedPollConsumeCeiling/4/real_time"
        "|BM_ChannelTypedPODFanoutTwoDispatchConsumeRetire/4/real_time"
        "|BM_OutputTypedPODNoopBinding"
        "|BM_ChannelInlineTypedPODPublish/manual_time"
        "|BM_ChannelInlineTypedPODPublishNoEvent/manual_time"
        "|BM_ChannelInlineTypedPODPublishActivePolling/manual_time"
        "|BM_ChannelInlineTypedPODPublishDeferredReceive/manual_time"
        "|BM_ChannelInlineTypedPODPublishAsyncWaiter/manual_time"
        "|BM_ChannelTypedPODSourceCut/manual_time"
        "|BM_ChannelTypedPODConsumeRetire/manual_time"
    )

    add_test(
        NAME BENCHMARK_RUNTIME_DISPATCHER
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_dispatcher_benchmark>"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_DISPATCHER
        PROPERTIES
            LABELS benchmark
    )

    add_test(
        NAME BENCHMARK_RUNTIME_DISPATCHER_MANAGED_SIGNALS
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_dispatcher_benchmark>"
                "--benchmark_filter=^BM_RunSession(OneDirectActivation|SteadyDirectActivations)(PersistentManagedLifecycle|TransientManagedDefault)?(/real_time)?$"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_DISPATCHER_MANAGED_SIGNALS
        PROPERTIES
            LABELS "benchmark;runtime;signals"
    )

    add_test(
        NAME BENCHMARK_RUNTIME_CHANNEL
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_channel_benchmark>"
                "--benchmark_filter=${RUNTIME_CHANNEL_SMOKE_FILTER}"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_CHANNEL
        PROPERTIES
            LABELS benchmark
    )

    add_test(
        NAME BENCHMARK_RUNTIME_PAYLOAD
        COMMAND ${Python3_EXECUTABLE}
                "${HOLOSCAN_BENCHMARK_RUNNER}"
                "$<TARGET_FILE:runtime_payload_benchmark>"
                "--benchmark_filter=BM_CompiledCatalog(CustomPod|Time|Imu)$"
                --benchmark_min_time=0.0001s
    )
    set_tests_properties(BENCHMARK_RUNTIME_PAYLOAD
        PROPERTIES
            LABELS benchmark
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH
        COMMAND runtime_graph_benchmark
                --workload latency
                --samples 16
                --warmup 4
                --publisher-period-us 1000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_REJECTS_AMBIGUOUS_FANOUT_SINK_CPU
        COMMAND ${CMAKE_COMMAND}
                -DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>
                -P "${CMAKE_CURRENT_SOURCE_DIR}/verify_ambiguous_fanout_sink_cpu_failure.cmake"
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_REJECTS_AMBIGUOUS_FANOUT_SINK_CPU
        PROPERTIES
            LABELS "benchmark;runtime;argument-validation"
            TIMEOUT 5
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SATURATED
        COMMAND runtime_graph_benchmark
                --workload saturated
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SATURATED
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    foreach(LOSSY_DEPTH IN ITEMS 1 2)
        if(LOSSY_DEPTH EQUAL 1)
            set(LOSSY_DEPTH_SUFFIX _DEPTH_ONE)
        else()
            set(LOSSY_DEPTH_SUFFIX)
        endif()
        foreach(LOSSY_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(LOSSY_DELIVERY_ARGUMENTS)
            if(LOSSY_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND LOSSY_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
            endif()
            set(LOSSY_DIAGNOSTIC_ARGUMENTS)
            if(LOSSY_DEPTH EQUAL 1 AND LOSSY_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND LOSSY_DIAGNOSTIC_ARGUMENTS --runtime-phase-counters)
            endif()
            set(
                LOSSY_TEST_NAME
                BENCHMARK_RUNTIME_GRAPH_DROP_OLDEST${LOSSY_DEPTH_SUFFIX}_${LOSSY_DELIVERY}
            )
            add_test(
                NAME ${LOSSY_TEST_NAME}
                COMMAND runtime_graph_benchmark
                        --workload saturated
                        --scenario pod
                        --queue-policy drop_oldest
                        --queue-depth ${LOSSY_DEPTH}
                        --notification-capacity ${LOSSY_DEPTH}
                        --samples 16
                        --warmup 4
                        --payload-validation all
                        ${LOSSY_DELIVERY_ARGUMENTS}
                        ${LOSSY_DIAGNOSTIC_ARGUMENTS}
                        --timeout-ms 10000
            )
            set_tests_properties(
                ${LOSSY_TEST_NAME}
                PROPERTIES
                    LABELS "benchmark;runtime;lossy"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(DROP_OLDEST_SCENARIO IN ITEMS
        schema
        third_party_schema
        tensor
        image
        multi_tensor_loan
        multi_tensor_staging
    )
        string(TOUPPER "${DROP_OLDEST_SCENARIO}" DROP_OLDEST_SCENARIO_UPPER)
        set(DROP_OLDEST_PAYLOAD_ARGUMENTS)
        set(DROP_OLDEST_WARMUP 4)
        if(NOT DROP_OLDEST_SCENARIO STREQUAL "schema" AND
           NOT DROP_OLDEST_SCENARIO STREQUAL "third_party_schema")
            list(APPEND DROP_OLDEST_PAYLOAD_ARGUMENTS
                --tensor-storage pageable_host
                --tensor-bytes 64
            )
        endif()
        if(DROP_OLDEST_SCENARIO STREQUAL "multi_tensor_loan")
            # Exercise the compiled hot-path field inventory before measured loan publications.
            set(DROP_OLDEST_WARMUP 8)
        endif()
        foreach(DROP_OLDEST_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(DROP_OLDEST_DELIVERY_ARGUMENTS)
            if(DROP_OLDEST_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND
                    DROP_OLDEST_DELIVERY_ARGUMENTS
                    --force-holomq-process-local-inline
                )
            endif()
            add_test(
                NAME
                    BENCHMARK_RUNTIME_GRAPH_DROP_OLDEST_${DROP_OLDEST_SCENARIO_UPPER}_${DROP_OLDEST_DELIVERY}
                COMMAND runtime_graph_benchmark
                        --workload saturated
                        --scenario ${DROP_OLDEST_SCENARIO}
                        --queue-policy drop_oldest
                        --queue-depth 2
                        --notification-capacity 2
                        --samples 16
                        --warmup ${DROP_OLDEST_WARMUP}
                        --payload-validation all
                        ${DROP_OLDEST_PAYLOAD_ARGUMENTS}
                        ${DROP_OLDEST_DELIVERY_ARGUMENTS}
                        --timeout-ms 10000
            )
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_DROP_OLDEST_${DROP_OLDEST_SCENARIO_UPPER}_${DROP_OLDEST_DELIVERY}
                PROPERTIES
                    LABELS "benchmark;runtime;lossy;${DROP_OLDEST_SCENARIO}"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(SATURATED_CAPACITY IN ITEMS 1 4 8)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_SATURATED_CAPACITY_${SATURATED_CAPACITY}
            COMMAND runtime_graph_benchmark
                    --workload saturated
                    --samples 16
                    --warmup 4
                    --queue-depth ${SATURATED_CAPACITY}
                    --notification-capacity ${SATURATED_CAPACITY}
                    --payload-validation all
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SATURATED_CAPACITY_${SATURATED_CAPACITY}
            PROPERTIES
                LABELS "benchmark;runtime;capacity"
                TIMEOUT 15
        )
    endforeach()

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY
        COMMAND runtime_graph_benchmark
                --workload notification_latency
                --samples 4
                --warmup 1
                --publisher-period-us 2000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    # Exercise the existing public placement contract with both Operators on one serial lane.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY_SHARED_LANE
        COMMAND runtime_graph_benchmark
                --workload notification_latency
                --scenario pod
                --operator-lanes shared
                --samples 4
                --warmup 1
                --publisher-period-us 2000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_NOTIFICATION_LATENCY_SHARED_LANE
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_NOTIFICATION_LATENCY
        COMMAND runtime_graph_benchmark
                --workload notification_latency
                --scenario source_only
                --samples 4
                --warmup 1
                --publisher-period-us 2000
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_NOTIFICATION_LATENCY
        PROPERTIES
            LABELS "benchmark;runtime"
            PASS_REGULAR_EXPRESSION
                "\"callback_thread_evidence\":\\{\"enabled\":true,\"valid\":true"
            TIMEOUT 15
    )

    # Bound notification pressure by completed source activations without a Channel edge.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_SATURATED
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario source_only
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_SATURATED
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SIGNAL
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario signal
                --samples 16
                --warmup 4
                --publisher-period-us 1000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SIGNAL
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    foreach(CHAIN_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
        set(CHAIN_DELIVERY_ARGUMENTS)
        if(CHAIN_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
            list(APPEND CHAIN_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
        endif()
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_LINEAR_CHAIN_${CHAIN_DELIVERY}
            COMMAND runtime_graph_benchmark
                    --workload latency
                    --scenario sample_forward
                    --relay-count 4
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --publisher-period-us 1000
                    --payload-validation all
                    ${CHAIN_DELIVERY_ARGUMENTS}
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_LINEAR_CHAIN_${CHAIN_DELIVERY}
            PROPERTIES
                LABELS "benchmark;runtime"
                TIMEOUT 15
        )
    endforeach()

    # Exercise the manually authored source/relay/sink partition cover under bounded downstream
    # pressure. Relay publication must preserve the owning Sample while retrying transient
    # backpressure instead of terminating the RunSession.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_BACKPRESSURE
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario sample_forward
                --relay-count 4
                --samples 32
                --warmup 8
                --publisher-period-us 200
                --queue-depth 1
                --notification-capacity 1
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_BACKPRESSURE
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 15
    )

    # A failing downstream callback must terminate a relay that is retrying a retained Sample
    # instead of leaving the benchmark process spinning until CTest kills it.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_FAILURE_IS_BOUNDED
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario sample_forward
                --relay-count 1
                --samples 8
                --warmup 0
                --publisher-period-us 1
                --queue-depth 1
                --notification-capacity 1
                --payload-validation none
                --test-fail-sink-after-relay-pressure
                --timeout-ms 1000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_PARTITIONED_CHAIN_FAILURE_IS_BOUNDED
        PROPERTIES
            LABELS "benchmark;runtime"
            TIMEOUT 5
            WILL_FAIL TRUE
            PASS_REGULAR_EXPRESSION "error=22"
            FAIL_REGULAR_EXPRESSION "runtime Graph benchmark timeout"
    )

    # Complete each logical sample only after all four transport-backed sinks receive it.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario pod
                --fanout 4
                --operator-lanes shared
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_POD_FANOUT
        PROPERTIES
            LABELS "benchmark;runtime;fanout"
            TIMEOUT 15
    )

    foreach(POD_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_FANOUT_WAKE_ARGUMENTS)
        if(POD_FANOUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-subscriber shared-lane POD service with compiler-selected
        # eventless delivery and with the explicit retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_EVENT_WAKE_${POD_FANOUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies exact two-recipient delivery and the requested
        # service-wide event-wake state without changing active fanout geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_EVENT_WAKE_${POD_FANOUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;event-wake"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: retain the Channel pump on an otherwise pump-free shared-lane eventless POD
    # HoloMQ row without selecting the independent forced-event-wake control.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_CHANNEL_PUMP_CONTROL
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario pod
                --operator-lanes shared
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --queue-depth 1
                --payload-validation all
                --force-holomq-process-local-inline
                --force-process-local-channel-pump
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: the runner requires event_wake=false, HoloMQ delivery, the explicit pump
    # provenance, and one entered Channel-pump support-thread role.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_CHANNEL_PUMP_CONTROL
        PROPERTIES
            LABELS "benchmark;runtime;holomq;eventless;channel-pump"
            TIMEOUT 15
    )

    foreach(POD_MULTI_OUTPUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_MULTI_OUTPUT_WAKE_ARGUMENTS)
        if(POD_MULTI_OUTPUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_MULTI_OUTPUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-output/one-subscriber-per-output shared-lane POD graph
        # with compiler-selected eventless delivery and with the exact retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_EVENT_WAKE_${POD_MULTI_OUTPUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario pod
                    --fanout 2
                    --multi-output
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_MULTI_OUTPUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies two HoloMQ services, one delivery from each output,
        # and the requested service-wide event-wake state without changing graph geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_EVENT_WAKE_${POD_MULTI_OUTPUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;multi-output;holomq;event-wake"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_MULTI_INPUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_MULTI_INPUT_WAKE_ARGUMENTS)
        if(POD_MULTI_INPUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_MULTI_INPUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-output/one-two-input-sink shared-lane POD graph with
        # compiler-selected eventless delivery and with the exact retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_INPUT_EVENT_WAKE_${POD_MULTI_INPUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario pod
                    --fanout 2
                    --multi-input
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_MULTI_INPUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies two HoloMQ services, two paired input reads per sink
        # callback, and the requested event-wake state without changing graph geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_MULTI_INPUT_EVENT_WAKE_${POD_MULTI_INPUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;multi-input;holomq;event-wake"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: run the two-output topology under sustained notification pressure with equal
    # explicit capacity on both independently published services.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_SATURATED_UNIFORM_DEPTH
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload saturated
                --scenario pod
                --fanout 2
                --multi-output
                --operator-lanes shared
                --samples 16
                --warmup 4
                --queue-depth 2
                --sink-queue-depths 2,2
                --notification-capacity 2
                --payload-validation all
                --force-holomq-process-local-inline
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: paired publication remains complete under pressure when both service queues
    # have equivalent geometry.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_SATURATED_UNIFORM_DEPTH
        PROPERTIES
            LABELS "benchmark;runtime;multi-output;holomq;queue-depth"
            TIMEOUT 15
    )

    # Key setup: pass heterogeneous service depths directly to the native parser for a
    # non-atomic paired-output topology.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_MIXED_DEPTH_REJECTED
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario pod
                --fanout 2
                --multi-output
                --operator-lanes shared
                --samples 16
                --warmup 4
                --queue-depth 4
                --sink-queue-depths 4,2
                --notification-capacity 4
                --payload-validation all
                --force-holomq-process-local-inline
                --timeout-ms 10000
    )
    # Key assertion: native validation rejects geometry that can partially commit the first
    # physical publication before the second service reports backpressure.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_MULTI_OUTPUT_MIXED_DEPTH_REJECTED
        PROPERTIES
            LABELS "benchmark;runtime;multi-output;holomq;queue-depth"
            WILL_FAIL TRUE
            TIMEOUT 15
    )

    foreach(POD_MIXED_POLICY_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_MIXED_POLICY_WAKE_ARGUMENTS)
        if(POD_MIXED_POLICY_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_MIXED_POLICY_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run one output through separate capacity-one blocking and drop-oldest HoloMQ
        # services on a shared lane with eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_MIXED_POLICY_EVENT_WAKE_${POD_MIXED_POLICY_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --mixed-policy-fanout
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --queue-depth 1
                    --notification-capacity 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_MIXED_POLICY_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies exact per-leg policies and delivery counts while the
        # pair changes only compiler-selected event wake and pump ownership.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_MIXED_POLICY_EVENT_WAKE_${POD_MIXED_POLICY_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;mixed-policy;event-wake"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_FANOUT_DEEP_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_FANOUT_DEEP_WAKE_ARGUMENTS)
        if(POD_FANOUT_DEEP_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_DEEP_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same depth-eight, two-subscriber shared-lane POD service with
        # compiler-selected eventless delivery and with the explicit retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DEPTH_8_EVENT_WAKE_${POD_FANOUT_DEEP_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --queue-depth 8
                    --notification-capacity 8
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_DEEP_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner proves exact two-recipient delivery and the requested
        # service-wide wake state without changing active fanout or queue capacity.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DEPTH_8_EVENT_WAKE_${POD_FANOUT_DEEP_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;event-wake;queue-depth"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_FANOUT_MIXED_DEPTH_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_FANOUT_MIXED_DEPTH_WAKE_ARGUMENTS)
        if(POD_FANOUT_MIXED_DEPTH_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_MIXED_DEPTH_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run one depth-four and one depth-two shared-lane POD subscriber in the same
        # compiler-owned service with eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_MIXED_DEPTH_EVENT_WAKE_${POD_FANOUT_MIXED_DEPTH_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --samples 16
                    --warmup 4
                    --queue-depth 4
                    --sink-queue-depths 4,2
                    --notification-capacity 4
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_MIXED_DEPTH_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies ordered 4/2 subscriber geometry, exact delivery and
        # final sequences, readiness, and the requested service-wide event-wake state.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_MIXED_DEPTH_EVENT_WAKE_${POD_FANOUT_MIXED_DEPTH_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;event-wake;queue-depth"
                TIMEOUT 15
        )
    endforeach()

    foreach(POD_FANOUT_DROP_OLDEST_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(POD_FANOUT_DROP_OLDEST_WAKE_ARGUMENTS)
        if(POD_FANOUT_DROP_OLDEST_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND POD_FANOUT_DROP_OLDEST_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: clock four-publication POD bursts into two depth-two drop-oldest subscribers
        # at the same 20 kpublication/s average rate with eventless and retained-wake delivery.
        # Use 16,384 samples (4,096 four-publication bursts), so the
        # accepted-offer measurement spans 4,095 inter-burst periods (819 ms) and
        # averages ordinary CI scheduler jitter while retaining the 2% rate gate.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DROP_OLDEST_EVENT_WAKE_${POD_FANOUT_DROP_OLDEST_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload saturated
                    --scenario pod
                    --fanout 2
                    --operator-lanes shared
                    --queue-policy drop_oldest
                    --queue-depth 2
                    --offered-rate-hz 20000
                    --offered-burst-size 4
                    --samples 16384
                    --warmup 15
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${POD_FANOUT_DROP_OLDEST_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner requires the configured clock rate, nonzero exact replacement
        # accounting on both legs, final-sequence delivery, and the requested service wake state.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_DROP_OLDEST_EVENT_WAKE_${POD_FANOUT_DROP_OLDEST_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;drop-oldest;event-wake"
                TIMEOUT 15
        )
    endforeach()

    # Prove fanout can materialize one independently accepted execution lane per subscriber.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_PER_SINK_LANES
        COMMAND runtime_graph_benchmark
                --workload saturated
                --scenario pod
                --fanout 4
                --operator-lanes per_sink
                --samples 16
                --warmup 4
                --queue-depth 2
                --notification-capacity 2
                --payload-validation all
                --runtime-phase-counters
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_POD_FANOUT_PER_SINK_LANES
        PROPERTIES
            LABELS "benchmark;runtime;fanout;placement"
            TIMEOUT 15
    )

    foreach(SCHEMA_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
        set(SCHEMA_DELIVERY_ARGUMENTS)
        if(SCHEMA_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
            list(APPEND SCHEMA_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
        endif()
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_SCHEMA_${SCHEMA_DELIVERY}
            COMMAND runtime_graph_benchmark
                    --workload latency
                    --scenario schema
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    ${SCHEMA_DELIVERY_ARGUMENTS}
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SCHEMA_${SCHEMA_DELIVERY}
            PROPERTIES
                LABELS "benchmark;runtime;schema"
                TIMEOUT 15
        )
    endforeach()

    foreach(SCHEMA_FANOUT_WAKE_SCENARIO IN ITEMS schema third_party_schema)
        string(TOUPPER "${SCHEMA_FANOUT_WAKE_SCENARIO}" SCHEMA_FANOUT_WAKE_SCENARIO_UPPER)
        foreach(SCHEMA_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(SCHEMA_FANOUT_WAKE_ARGUMENTS)
            if(SCHEMA_FANOUT_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND SCHEMA_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: run the same two-subscriber SDK/third-party schema service with
            # compiler-selected eventless delivery and with the retained-wake control.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_${SCHEMA_FANOUT_WAKE_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${SCHEMA_FANOUT_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario "${SCHEMA_FANOUT_WAKE_SCENARIO}"
                        --fanout 2
                        --operator-lanes shared
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${SCHEMA_FANOUT_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies exact two-recipient schema delivery and the
            # requested service-wide wake state without changing active fanout geometry.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_${SCHEMA_FANOUT_WAKE_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${SCHEMA_FANOUT_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;schema;holomq;event-wake"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(IMAGE_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
        set(IMAGE_DELIVERY_ARGUMENTS)
        if(IMAGE_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
            list(APPEND IMAGE_DELIVERY_ARGUMENTS --force-holomq-process-local-inline)
        endif()
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_${IMAGE_DELIVERY}
            COMMAND runtime_graph_benchmark
                    --workload latency
                    --scenario image
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    ${IMAGE_DELIVERY_ARGUMENTS}
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_IMAGE_${IMAGE_DELIVERY}
            PROPERTIES
                LABELS "benchmark;runtime;schema;tensor"
                TIMEOUT 15
        )
    endforeach()

    foreach(MULTI_TENSOR_SCENARIO IN ITEMS multi_tensor_loan multi_tensor_staging)
        string(TOUPPER "${MULTI_TENSOR_SCENARIO}" MULTI_TENSOR_SCENARIO_UPPER)
        set(MULTI_TENSOR_WARMUP 2)
        if(MULTI_TENSOR_SCENARIO STREQUAL "multi_tensor_loan")
            set(MULTI_TENSOR_WARMUP 8)
        endif()
        foreach(MULTI_TENSOR_DELIVERY IN ITEMS DIRECT HOLOMQ_FALLBACK)
            set(MULTI_TENSOR_DELIVERY_ARGUMENTS)
            if(MULTI_TENSOR_DELIVERY STREQUAL "HOLOMQ_FALLBACK")
                list(APPEND MULTI_TENSOR_DELIVERY_ARGUMENTS
                    --force-holomq-process-local-inline)
            endif()
            foreach(MULTI_TENSOR_VALIDATION IN ITEMS ALL FINAL)
                string(TOLOWER "${MULTI_TENSOR_VALIDATION}" MULTI_TENSOR_VALIDATION_LOWER)
                set(MULTI_TENSOR_TEST_NAME
                    BENCHMARK_RUNTIME_GRAPH_${MULTI_TENSOR_SCENARIO_UPPER}_${MULTI_TENSOR_DELIVERY}_${MULTI_TENSOR_VALIDATION})
                add_test(
                    NAME ${MULTI_TENSOR_TEST_NAME}
                    COMMAND runtime_graph_benchmark
                            --workload latency
                            --scenario "${MULTI_TENSOR_SCENARIO}"
                            --tensor-storage pageable_host
                            --tensor-bytes 64
                            --samples 8
                            --warmup ${MULTI_TENSOR_WARMUP}
                            --publisher-period-us 1000
                            --payload-validation "${MULTI_TENSOR_VALIDATION_LOWER}"
                            ${MULTI_TENSOR_DELIVERY_ARGUMENTS}
                            --timeout-ms 10000
                )
                set_tests_properties(
                    ${MULTI_TENSOR_TEST_NAME}
                    PROPERTIES
                        LABELS "benchmark;runtime;schema;tensor"
                        TIMEOUT 15
                )
            endforeach()
        endforeach()
    endforeach()

    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_LOAN_UNDERSIZED_WARMUP_REJECTED
        COMMAND ${CMAKE_COMMAND}
                "-DHOLOSCAN_RUNTIME_GRAPH_BENCHMARK=$<TARGET_FILE:runtime_graph_benchmark>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_undersized_loan_warmup.cmake"
    )
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_LOAN_UNDERSIZED_WARMUP_REJECTED
        PROPERTIES
            LABELS "benchmark;runtime;schema;tensor;negative"
            TIMEOUT 15
    )

    # Key setup: exercise each one-deep shared-lane direct-retained Tensor payload shape.
    foreach(TENSOR_ORDINARY_DISPATCH_SCENARIO IN ITEMS
        tensor
        image
        multi_tensor_loan
        multi_tensor_staging
    )
        string(TOUPPER
            "${TENSOR_ORDINARY_DISPATCH_SCENARIO}"
            TENSOR_ORDINARY_DISPATCH_SCENARIO_UPPER
        )
        set(TENSOR_ORDINARY_DISPATCH_BULK_WORK_ARGUMENTS)
        set(TENSOR_ORDINARY_DISPATCH_WARMUP 2)
        if(TENSOR_ORDINARY_DISPATCH_SCENARIO STREQUAL "multi_tensor_loan")
            set(TENSOR_ORDINARY_DISPATCH_WARMUP 8)
        endif()
        if(NOT TENSOR_ORDINARY_DISPATCH_SCENARIO STREQUAL "tensor")
            list(APPEND
                TENSOR_ORDINARY_DISPATCH_BULK_WORK_ARGUMENTS
                --bulk-work-profile descriptor_only
            )
        endif()
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${TENSOR_ORDINARY_DISPATCH_SCENARIO_UPPER}_NO_CONTINUATION_RECEIPT
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${TENSOR_ORDINARY_DISPATCH_SCENARIO}"
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup ${TENSOR_ORDINARY_DISPATCH_WARMUP}
                    --publisher-period-us 1000
                    --payload-validation all
                    ${TENSOR_ORDINARY_DISPATCH_BULK_WORK_ARGUMENTS}
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner requires ordinary dispatch with no continuation receipts.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${TENSOR_ORDINARY_DISPATCH_SCENARIO_UPPER}_NO_CONTINUATION_RECEIPT
            PROPERTIES
                LABELS "benchmark;runtime;tensor;ordinary-dispatch"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: exercise each certified non-Tensor 1:1 HoloMQ path on a shared lane.
    foreach(HOLOMQ_ORDINARY_DISPATCH_SCENARIO IN ITEMS pod schema third_party_schema)
        string(TOUPPER
            "${HOLOMQ_ORDINARY_DISPATCH_SCENARIO}"
            HOLOMQ_ORDINARY_DISPATCH_SCENARIO_UPPER
        )
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${HOLOMQ_ORDINARY_DISPATCH_SCENARIO_UPPER}_HOLOMQ_NO_CONTINUATION_RECEIPT
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${HOLOMQ_ORDINARY_DISPATCH_SCENARIO}"
                    --operator-lanes shared
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --payload-validation all
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner requires ordinary dispatch with no continuation receipts.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${HOLOMQ_ORDINARY_DISPATCH_SCENARIO_UPPER}_HOLOMQ_NO_CONTINUATION_RECEIPT
            PROPERTIES
                LABELS "benchmark;runtime;holomq;ordinary-dispatch"
                TIMEOUT 15
        )
    endforeach()

    # Key setup: keep the forced-HoloMQ POD producer and consumer on separate partitions.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_CONTROL
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario pod
                --operator-lanes separate
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --payload-validation all
                --force-holomq-process-local-inline
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: the runner requires event wake and ordinary dispatch for this control.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_CONTROL
        PROPERTIES
            LABELS "benchmark;runtime;holomq;event-wake"
            TIMEOUT 15
    )

    # Key setup: retain event wake on the one-deep shared-lane HoloMQ POD path.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_SAME_LANE_CONTROL
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload latency
                --scenario pod
                --operator-lanes shared
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --payload-validation all
                --force-holomq-process-local-inline
                --force-holomq-event-wake
                --runtime-phase-counters
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: the runner requires ordinary dispatch and an active event-wake service.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_POD_HOLOMQ_EVENT_WAKE_SAME_LANE_CONTROL
        PROPERTIES
            LABELS "benchmark;runtime;holomq;ordinary-dispatch;event-wake"
            TIMEOUT 15
    )

    foreach(TENSOR_WAKE_SCENARIO IN ITEMS
        tensor
        image
        multi_tensor_loan
        multi_tensor_staging
    )
        string(TOUPPER "${TENSOR_WAKE_SCENARIO}" TENSOR_WAKE_SCENARIO_UPPER)
        set(TENSOR_WAKE_BULK_WORK_ARGUMENTS)
        set(TENSOR_WAKE_WARMUP 2)
        if(TENSOR_WAKE_SCENARIO STREQUAL "multi_tensor_loan")
            set(TENSOR_WAKE_WARMUP 8)
        endif()
        if(NOT TENSOR_WAKE_SCENARIO STREQUAL "tensor")
            list(APPEND TENSOR_WAKE_BULK_WORK_ARGUMENTS --bulk-work-profile descriptor_only)
        endif()

        # Key setup: force an exact shared-lane Tensor payload edge through one-deep HoloMQ.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_SUPPRESSED
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${TENSOR_WAKE_SCENARIO}"
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup ${TENSOR_WAKE_WARMUP}
                    --publisher-period-us 1000
                    --payload-validation all
                    ${TENSOR_WAKE_BULK_WORK_ARGUMENTS}
                    --force-holomq-process-local-inline
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner requires compiler-selected event-wake suppression.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_SUPPRESSED
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;event-wake"
                TIMEOUT 15
        )

        # Key setup: select the private forced-wake compiler seam for the same Tensor graph.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_CONTROL
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario "${TENSOR_WAKE_SCENARIO}"
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup ${TENSOR_WAKE_WARMUP}
                    --publisher-period-us 1000
                    --payload-validation all
                    ${TENSOR_WAKE_BULK_WORK_ARGUMENTS}
                    --force-holomq-process-local-inline
                    --force-holomq-event-wake
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        # Key assertion: the runner requires the active event-wake service on the control row.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_${TENSOR_WAKE_SCENARIO_UPPER}_HOLOMQ_EVENT_WAKE_CONTROL
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;event-wake"
                TIMEOUT 15
        )
        if(TENSOR_WAKE_SCENARIO STREQUAL "tensor")
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_TENSOR_HOLOMQ_EVENT_WAKE_SUPPRESSED
                BENCHMARK_RUNTIME_GRAPH_TENSOR_HOLOMQ_EVENT_WAKE_CONTROL
                PROPERTIES
                    SKIP_RETURN_CODE 77
            )
        endif()
    endforeach()

    foreach(RAW_TENSOR_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(RAW_TENSOR_FANOUT_WAKE_ARGUMENTS)
        if(RAW_TENSOR_FANOUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND RAW_TENSOR_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-recipient shared-lane raw Tensor service with
        # compiler-selected eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_EVENT_WAKE_${RAW_TENSOR_FANOUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario tensor
                    --fanout 2
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${RAW_TENSOR_FANOUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies both Tensor recipients, pool/release geometry, and
        # the requested service-wide wake state without changing active fanout geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_EVENT_WAKE_${RAW_TENSOR_FANOUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;tensor;event-wake"
                SKIP_RETURN_CODE 77
                TIMEOUT 15
        )
    endforeach()

    foreach(RAW_TENSOR_FANOUT_WIDTH IN ITEMS 4 8)
        foreach(RAW_TENSOR_WIDE_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(RAW_TENSOR_WIDE_FANOUT_WAKE_ARGUMENTS)
            if(RAW_TENSOR_WIDE_FANOUT_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND RAW_TENSOR_WIDE_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: scale the same shared-lane raw Tensor service to four and eight active
            # recipients with compiler-selected eventless and retained-wake delivery.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_WIDTH}_EVENT_WAKE_${RAW_TENSOR_WIDE_FANOUT_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario tensor
                        --fanout "${RAW_TENSOR_FANOUT_WIDTH}"
                        --operator-lanes shared
                        --tensor-storage pageable_host
                        --tensor-bytes 64
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${RAW_TENSOR_WIDE_FANOUT_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies every recipient, pool/release geometry, pointer
            # identity, final payload, and the requested service-wide wake state.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_WIDTH}_EVENT_WAKE_${RAW_TENSOR_WIDE_FANOUT_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;event-wake"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(RAW_TENSOR_FANOUT_STORAGE IN ITEMS pinned_host device)
        string(TOUPPER "${RAW_TENSOR_FANOUT_STORAGE}" RAW_TENSOR_FANOUT_STORAGE_UPPER)
        foreach(RAW_TENSOR_STORAGE_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(RAW_TENSOR_STORAGE_WAKE_ARGUMENTS)
            if(RAW_TENSOR_STORAGE_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND RAW_TENSOR_STORAGE_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: run the same two-recipient shared-lane raw Tensor service with pinned-host
            # and device pool backing under eventless and retained-wake delivery.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_STORAGE_UPPER}_EVENT_WAKE_${RAW_TENSOR_STORAGE_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario tensor
                        --fanout 2
                        --operator-lanes shared
                        --tensor-storage "${RAW_TENSOR_FANOUT_STORAGE}"
                        --tensor-bytes 64
                        --samples 8
                        --warmup 2
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${RAW_TENSOR_STORAGE_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies the requested storage, both recipients, pool and
            # release geometry, pointer identity, final payload, and service-wide wake state.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_FANOUT_${RAW_TENSOR_FANOUT_STORAGE_UPPER}_EVENT_WAKE_${RAW_TENSOR_STORAGE_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;storage;event-wake"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    # Key setup: sustain 1 MiB asynchronous device writes long enough to recycle the six-slot
    # raw Tensor pool while two shared-lane recipients retain each publication.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_DEVICE_POOL_RETRY
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload saturated
                --scenario tensor
                --fanout 2
                --operator-lanes shared
                --tensor-storage device
                --tensor-bytes 1048576
                --samples 2048
                --warmup 100
                --queue-depth 1
                --payload-validation final
                --repetitions 1
                --session-timeout-ms 10000
    )
    # Key assertion: transient device-pool exhaustion is retried until every publication reaches
    # both recipients with pointer identity and final payload validation intact.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_DEVICE_POOL_RETRY
        PROPERTIES
            LABELS "benchmark;runtime;fanout;tensor;storage;pool-retry"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )

    # Key setup: inject one allocation-boundary kResourceExhausted result that the
    # notification-driven raw-Tensor source consumes inside its first compute activation.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_IN_ACTIVATION_POOL_RETRY
        COMMAND $<TARGET_FILE:runtime_graph_benchmark>
                --workload saturated
                --scenario tensor
                --operator-lanes shared
                --samples 4
                --warmup 1
                --queue-depth 1
                --payload-validation all
                --runtime-phase-counters
                --test-source-tensor-pool-in-activation-retries 1
                --timeout-ms 10000
    )
    # Key assertion: the run completes six callbacks and reports the pool retry separately while
    # native phase validation and JSON retain zero Channel emit backpressure.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_RAW_TENSOR_IN_ACTIVATION_POOL_RETRY
        PROPERTIES
            LABELS "benchmark;runtime;tensor;pool-retry;phase-counters"
            PASS_REGULAR_EXPRESSION
                "\"source_callbacks\":6.*\"backpressured_emits\":0.*\"source_tensor_pool_in_activation_retries\":1"
            SKIP_RETURN_CODE 77
            TIMEOUT 15
    )

    # Key setup: inject one allocation-boundary kResourceExhausted result that the
    # notification-driven multi-Tensor source consumes inside its first compute activation.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_IN_ACTIVATION_POOL_RETRY
        COMMAND $<TARGET_FILE:runtime_graph_benchmark>
                --workload saturated
                --scenario multi_tensor
                --operator-lanes shared
                --samples 4
                --warmup 1
                --queue-depth 1
                --payload-validation all
                --runtime-phase-counters
                --test-source-tensor-pool-in-activation-retries 1
                --timeout-ms 10000
    )
    # Key assertion: the run completes six callbacks and reports the pool retry separately while
    # native phase validation and JSON retain zero Channel emit backpressure.
    set_tests_properties(
        BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_IN_ACTIVATION_POOL_RETRY
        PROPERTIES
            LABELS "benchmark;runtime;multi_tensor;pool-retry;phase-counters"
            PASS_REGULAR_EXPRESSION
                "\"source_callbacks\":6.*\"backpressured_emits\":0.*\"source_tensor_pool_in_activation_retries\":1"
            SKIP_RETURN_CODE 77
            TIMEOUT 15
    )

    foreach(IMAGE_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
        set(IMAGE_FANOUT_WAKE_ARGUMENTS)
        if(IMAGE_FANOUT_WAKE_MODE STREQUAL "CONTROL")
            list(APPEND IMAGE_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
        endif()
        # Key setup: run the same two-recipient shared-lane Image descriptor-sidecar service with
        # compiler-selected eventless delivery and with the retained-wake control.
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_FANOUT_EVENT_WAKE_${IMAGE_FANOUT_WAKE_MODE}
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario image
                    --fanout 2
                    --operator-lanes shared
                    --tensor-storage pageable_host
                    --tensor-bytes 64
                    --bulk-work-profile descriptor_only
                    --samples 8
                    --warmup 2
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation all
                    --runtime-phase-counters
                    --repetitions 1
                    --session-timeout-ms 10000
                    ${IMAGE_FANOUT_WAKE_ARGUMENTS}
        )
        # Key assertion: the runner verifies both Image envelopes, shared sidecar geometry, and
        # the requested service-wide wake state without changing active fanout geometry.
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_IMAGE_FANOUT_EVENT_WAKE_${IMAGE_FANOUT_WAKE_MODE}
            PROPERTIES
                LABELS "benchmark;runtime;fanout;holomq;tensor;schema;event-wake"
                TIMEOUT 15
        )
    endforeach()

    foreach(MULTI_TENSOR_FANOUT_SCENARIO IN ITEMS multi_tensor_loan multi_tensor_staging)
        string(TOUPPER
            "${MULTI_TENSOR_FANOUT_SCENARIO}"
            MULTI_TENSOR_FANOUT_SCENARIO_UPPER
        )
        set(MULTI_TENSOR_FANOUT_WARMUP 2)
        if(MULTI_TENSOR_FANOUT_SCENARIO STREQUAL "multi_tensor_loan")
            set(MULTI_TENSOR_FANOUT_WARMUP 8)
        endif()
        foreach(MULTI_TENSOR_FANOUT_WAKE_MODE IN ITEMS SUPPRESSED CONTROL)
            set(MULTI_TENSOR_FANOUT_WAKE_ARGUMENTS)
            if(MULTI_TENSOR_FANOUT_WAKE_MODE STREQUAL "CONTROL")
                list(APPEND MULTI_TENSOR_FANOUT_WAKE_ARGUMENTS --force-holomq-event-wake)
            endif()
            # Key setup: run the same two-recipient shared-lane multi-Tensor descriptor service
            # with compiler-selected eventless delivery and with the retained-wake control.
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_${MULTI_TENSOR_FANOUT_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${MULTI_TENSOR_FANOUT_WAKE_MODE}
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario "${MULTI_TENSOR_FANOUT_SCENARIO}"
                        --fanout 2
                        --operator-lanes shared
                        --tensor-storage pageable_host
                        --tensor-bytes 64
                        --bulk-work-profile descriptor_only
                        --samples 8
                        --warmup ${MULTI_TENSOR_FANOUT_WARMUP}
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation all
                        --runtime-phase-counters
                        --repetitions 1
                        --session-timeout-ms 10000
                        ${MULTI_TENSOR_FANOUT_WAKE_ARGUMENTS}
            )
            # Key assertion: the runner verifies both descriptor fields at both recipients,
            # pool/release geometry, and the requested service-wide wake state.
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_${MULTI_TENSOR_FANOUT_SCENARIO_UPPER}_FANOUT_EVENT_WAKE_${MULTI_TENSOR_FANOUT_WAKE_MODE}
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;schema;event-wake"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(TENSOR_STORAGE IN ITEMS pageable_host pinned_host device)
        string(TOUPPER "${TENSOR_STORAGE}" TENSOR_STORAGE_UPPER)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_TENSOR_${TENSOR_STORAGE_UPPER}
            COMMAND runtime_graph_benchmark
                    --workload latency
                    --scenario tensor
                    --tensor-storage "${TENSOR_STORAGE}"
                    --tensor-bytes 64
                    --samples 16
                    --warmup 4
                    --publisher-period-us 1000
                    --payload-validation final
                    --timeout-ms 10000
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_TENSOR_${TENSOR_STORAGE_UPPER}
            PROPERTIES
                LABELS "benchmark;runtime;tensor"
                SKIP_RETURN_CODE 77
                TIMEOUT 15
        )
    endforeach()

    # Prove that eventless HoloMQ descriptor workloads retain identical no-write publication
    # geometry across storage arms. Image and staging reuse immutable stream-aware external
    # backings so device rows exercise Runtime's ordinary Tensor lifetime contract.
    foreach(DESCRIPTOR_STORAGE IN ITEMS pageable_host pinned_host device)
        string(TOUPPER "${DESCRIPTOR_STORAGE}" DESCRIPTOR_STORAGE_UPPER)
        add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_IMAGE_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
                COMMAND ${Python3_EXECUTABLE}
                        "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                        "$<TARGET_FILE:runtime_graph_benchmark>"
                        --workload latency
                        --scenario image
                        --fanout 2
                        --operator-lanes shared
                        --tensor-storage "${DESCRIPTOR_STORAGE}"
                        --tensor-bytes 4096
                        --bulk-work-profile descriptor_only
                        --samples 8
                        --warmup 8
                        --publisher-period-us 1000
                        --queue-depth 1
                        --payload-validation final
                        --repetitions 1
                        --session-timeout-ms 10000
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_IMAGE_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;schema;storage"
                SKIP_RETURN_CODE 77
                TIMEOUT 30
        )

        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload latency
                    --scenario multi_tensor
                    --operator-lanes shared
                    --tensor-storage "${DESCRIPTOR_STORAGE}"
                    --tensor-bytes 4096
                    --bulk-work-profile descriptor_only
                    --samples 8
                    --warmup 8
                    --publisher-period-us 1000
                    --queue-depth 1
                    --payload-validation final
                    --force-holomq-process-local-inline
                    --repetitions 1
                    --session-timeout-ms 10000
        )
        set_tests_properties(
            BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
            PROPERTIES
                LABELS "benchmark;runtime;holomq;tensor;schema;multi_tensor;storage"
                SKIP_RETURN_CODE 77
                TIMEOUT 30
        )

        foreach(DESCRIPTOR_MODE IN ITEMS multi_tensor_loan multi_tensor_staging)
            string(TOUPPER "${DESCRIPTOR_MODE}" DESCRIPTOR_MODE_UPPER)
            set(DESCRIPTOR_MODE_DIAGNOSTIC_ARGUMENTS)
            if(DESCRIPTOR_MODE STREQUAL "multi_tensor_loan")
                # Preserve controller, lane, and channel progress if a runtime-owned field pool
                # fails to reach the first source callback before the session deadline.
                list(APPEND DESCRIPTOR_MODE_DIAGNOSTIC_ARGUMENTS --runtime-phase-counters)
            endif()
            add_test(
                    NAME BENCHMARK_RUNTIME_GRAPH_${DESCRIPTOR_MODE_UPPER}_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_benchmark>"
                            --workload latency
                            --scenario "${DESCRIPTOR_MODE}"
                            --fanout 2
                            --operator-lanes shared
                            --tensor-storage "${DESCRIPTOR_STORAGE}"
                            --tensor-bytes 4096
                            --bulk-work-profile descriptor_only
                            --samples 8
                            --warmup 8
                            --publisher-period-us 1000
                            --queue-depth 1
                            --payload-validation final
                            ${DESCRIPTOR_MODE_DIAGNOSTIC_ARGUMENTS}
                            --repetitions 1
                            --session-timeout-ms 10000
            )
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_${DESCRIPTOR_MODE_UPPER}_${DESCRIPTOR_STORAGE_UPPER}_STORAGE_PARITY_SMOKE
                PROPERTIES
                    LABELS "benchmark;runtime;fanout;holomq;tensor;schema;multi_tensor;storage"
                    SKIP_RETURN_CODE 77
                    TIMEOUT 30
            )
        endforeach()
    endforeach()

    # Prove the docs-oriented raw Tensor profile initializes every compiler-owned pool slot before
    # measurement and performs descriptor-only publications on the hot path.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_TENSOR_PINNED_HOST_DESCRIPTOR_ONLY
        COMMAND ${Python3_EXECUTABLE}
                "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                "$<TARGET_FILE:runtime_graph_benchmark>"
                --workload saturated
                --scenario tensor
                --tensor-storage pinned_host
                --tensor-bytes 4096
                --bulk-work-profile descriptor_only
                --samples 16
                --warmup 8
                --queue-depth 2
                --notification-capacity 2
                --payload-validation final
                --session-timeout-ms 10000
                --process-timeout-sec 15
                --repetitions 1
                --output-dir
                "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_tensor_pinned_host_descriptor_only"
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_TENSOR_PINNED_HOST_DESCRIPTOR_ONLY
        PROPERTIES
            LABELS "benchmark;runtime;tensor"
            SKIP_RETURN_CODE 77
            TIMEOUT 30
    )

    # Exercise both paced latency and saturated throughput for the generated native type with two
    # independently pooled Tensor fields used by the Core Runtime docs table.
    foreach(MULTI_TENSOR_WORKLOAD IN ITEMS latency saturated)
        string(TOUPPER "${MULTI_TENSOR_WORKLOAD}" MULTI_TENSOR_WORKLOAD_UPPER)
        if(MULTI_TENSOR_WORKLOAD STREQUAL "latency")
            set(MULTI_TENSOR_QUEUE_ARGUMENTS --queue-depth 1 --publisher-period-us 1000)
        else()
            set(MULTI_TENSOR_QUEUE_ARGUMENTS --queue-depth 2 --notification-capacity 2)
        endif()
        add_test(
            NAME "BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${MULTI_TENSOR_WORKLOAD_UPPER}"
            COMMAND ${Python3_EXECUTABLE}
                    "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_benchmark.py"
                    "$<TARGET_FILE:runtime_graph_benchmark>"
                    --workload "${MULTI_TENSOR_WORKLOAD}"
                    --scenario multi_tensor
                    --tensor-storage pageable_host
                    --tensor-bytes 4096
                    --bulk-work-profile descriptor_only
                    --samples 8
                    --warmup 8
                    ${MULTI_TENSOR_QUEUE_ARGUMENTS}
                    --payload-validation final
                    --session-timeout-ms 10000
                    --process-timeout-sec 15
                    --repetitions 1
                    --output-dir
                    "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_multi_tensor_${MULTI_TENSOR_WORKLOAD}"
        )
        set_tests_properties(
            "BENCHMARK_RUNTIME_GRAPH_MULTI_TENSOR_${MULTI_TENSOR_WORKLOAD_UPPER}"
            PROPERTIES
                LABELS "benchmark;runtime;tensor;schema;multi_tensor"
                TIMEOUT 30
        )
    endforeach()

    # Preserve one transport-backed Tensor control beside the default direct-retained rows.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_TENSOR_PAGEABLE_HOST_HOLOMQ_FALLBACK
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario tensor
                --tensor-storage pageable_host
                --tensor-bytes 64
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --payload-validation all
                --force-holomq-process-local-inline
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_TENSOR_PAGEABLE_HOST_HOLOMQ_FALLBACK
        PROPERTIES
            LABELS "benchmark;runtime;tensor"
            TIMEOUT 15
    )
endif()
