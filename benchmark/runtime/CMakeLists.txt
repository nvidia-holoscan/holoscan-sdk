# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(runtime_dispatcher_benchmark
    runtime_dispatcher_benchmark.cpp
    ${CMAKE_SOURCE_DIR}/tests/support/direct_port_binding.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/dispatcher.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/physical_readiness_binding.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduling/deterministic_serial_ready_policy.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/scheduling/ready_policy_catalog.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/execution_lane.cpp
    ${CMAKE_SOURCE_DIR}/src/threading/linux_thread_controller.cpp
)
add_dependencies(holoscan-benchmark-targets runtime_dispatcher_benchmark)
target_include_directories(runtime_dispatcher_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
)
target_link_libraries(runtime_dispatcher_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
)

add_executable(runtime_channel_benchmark
    runtime_channel_benchmark.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/process_local_channel_plan_validation.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/endpoint_clock_health_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_channel_binding.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/execution_lane.cpp
    ${CMAKE_SOURCE_DIR}/src/threading/linux_thread_controller.cpp
)
add_dependencies(holoscan-benchmark-targets runtime_channel_benchmark)
target_include_directories(runtime_channel_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
)
target_link_libraries(runtime_channel_benchmark
    PRIVATE
        benchmark::benchmark_main
        holoscan::core
        Threads::Threads
)

add_executable(runtime_payload_benchmark
    runtime_payload_benchmark.cpp
    ${CMAKE_SOURCE_DIR}/src/compiler/process_local_channel_plan_validation.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/endpoint_clock_health.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/endpoint_clock_health_runtime.cpp
    ${CMAKE_SOURCE_DIR}/src/runtime/runtime_channel_binding.cpp
    ${CMAKE_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_payload_benchmark holoscan_flatbuffer_schemas_generated)
add_dependencies(holoscan-benchmark-targets runtime_payload_benchmark)
target_include_directories(runtime_payload_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
)
target_link_libraries(runtime_payload_benchmark
    PRIVATE
        benchmark::benchmark_main
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
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

add_executable(runtime_graph_benchmark
    runtime_graph_benchmark.cpp
    runtime_graph_benchmark_common.hpp
    ../common/transport_benchmark_common.hpp
    ${CMAKE_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_graph_benchmark
    holoscan_flatbuffer_schemas_generated
    runtime_multi_tensor_benchmark_generated
)
add_dependencies(holoscan-benchmark-targets runtime_graph_benchmark)
target_include_directories(runtime_graph_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
)
target_link_libraries(runtime_graph_benchmark
    PRIVATE
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        Threads::Threads
)

add_executable(runtime_graph_interprocess_benchmark
    runtime_graph_interprocess_benchmark.cpp
    runtime_graph_benchmark_common.hpp
    ../common/transport_benchmark_common.hpp
    ${CMAKE_SOURCE_DIR}/tests/common/allocation_counter.cpp
)
add_dependencies(runtime_graph_interprocess_benchmark
    holoscan_flatbuffer_schemas_generated
    runtime_multi_tensor_benchmark_generated
)
add_dependencies(holoscan-benchmark-targets runtime_graph_interprocess_benchmark)
target_include_directories(runtime_graph_interprocess_benchmark
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests
        "${HOLOSCAN_SCHEMA_GENERATED_DIR}"
        "${RUNTIME_MULTI_TENSOR_BENCHMARK_GENERATED_DIR}"
)
target_link_libraries(runtime_graph_interprocess_benchmark
    PRIVATE
        flatbuffers::flatbuffers
        holoscan::core
        holoscan::schema
        Threads::Threads
)

if(BUILD_TESTING)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME BENCHMARK_RUNTIME_GRAPH_RUNNER_PYTHON_TESTS
            COMMAND ${Python3_EXECUTABLE} -m pytest -q
                    "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_graph_benchmark.py"
        )
        set_tests_properties(BENCHMARK_RUNTIME_GRAPH_RUNNER_PYTHON_TESTS
            PROPERTIES
                LABELS "benchmark;ea1;runtime"
        )

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
                LABELS "benchmark;ea1;runtime"
                TIMEOUT 45
        )

        if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
            add_test(
                NAME BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_RUNNER_PYTHON_TESTS
                COMMAND ${Python3_EXECUTABLE} -m pytest -q
                        "${CMAKE_CURRENT_SOURCE_DIR}/test_run_runtime_graph_interprocess_benchmark.py"
            )
            set_tests_properties(BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_RUNNER_PYTHON_TESTS
                PROPERTIES
                    LABELS "benchmark;ea1;runtime;interprocess"
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
                    LABELS "benchmark;ea1;runtime;interprocess"
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
                        LABELS "benchmark;ea1;runtime;interprocess;tensor"
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
                        LABELS "benchmark;ea1;runtime;interprocess;tensor"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
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
                        LABELS "benchmark;ea1;runtime;interprocess;tensor;schema"
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
                        LABELS "benchmark;ea1;runtime;interprocess;tensor;schema"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
            endforeach()

            # Exercise both multi-Tensor authoring paths through real Operator callbacks. The
            # excluded final warmup sample proves the direct bundle and staged-value callbacks are
            # allocation-free when the sink uses the activation-scoped borrowed batch view.
            foreach(multi_tensor_payload IN ITEMS multi_tensor_loan multi_tensor_staging)
                string(TOUPPER "${multi_tensor_payload}" multi_tensor_payload_upper)
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload "${multi_tensor_payload}"
                            --tensor-storage pageable_host
                            --tensor-bytes 4096
                            --bulk-work-profile descriptor_only
                            --workload latency
                            --samples 4
                            --warmup 8
                            --queue-depth 1
                            --publisher-period-us 1000
                            --payload-validation all
                            --session-timeout-ms 10000
                            --process-timeout-sec 30
                            --output-dir
                            "${CMAKE_CURRENT_BINARY_DIR}/runtime_graph_interprocess_${multi_tensor_payload}_smoke"
                )
                set_tests_properties(
                    "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_SMOKE"
                    PROPERTIES
                        LABELS "benchmark;ea1;runtime;interprocess;tensor;schema;multi_tensor"
                        SKIP_RETURN_CODE 77
                        TIMEOUT 120
                )
                # Saturation exhausts the multi-field reservation metadata before the next
                # envelope loan; keep this case to prevent reintroducing that progress cycle.
                add_test(
                    NAME "BENCHMARK_RUNTIME_GRAPH_INTERPROCESS_${multi_tensor_payload_upper}_SATURATED_SMOKE"
                    COMMAND ${Python3_EXECUTABLE}
                            "${CMAKE_CURRENT_SOURCE_DIR}/run_runtime_graph_interprocess_benchmark.py"
                            "$<TARGET_FILE:runtime_graph_interprocess_benchmark>"
                            --payload "${multi_tensor_payload}"
                            --tensor-storage pageable_host
                            --tensor-bytes 4096
                            --bulk-work-profile descriptor_only
                            --workload saturated
                            --samples 16
                            --warmup 8
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
                        LABELS "benchmark;ea1;runtime;interprocess;tensor;schema;multi_tensor;saturated"
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
                        LABELS "benchmark;ea1;runtime;interprocess;schema"
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
                    LABELS "benchmark;ea1;runtime;interprocess;fanout"
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
                    LABELS "benchmark;ea1;runtime;interprocess;fanout"
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
                    LABELS "benchmark;ea1;runtime;interprocess;chain"
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
                    LABELS "benchmark;ea1;runtime;interprocess"
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
                LABELS "benchmark;ea1;runtime"
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
                LABELS "benchmark;ea1;runtime;fanout;placement"
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
            LABELS "benchmark;ea1;runtime"
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
            LABELS "benchmark;ea1;runtime"
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
                        --timeout-ms 10000
            )
            set_tests_properties(
                ${LOSSY_TEST_NAME}
                PROPERTIES
                    LABELS "benchmark;ea1;runtime;lossy"
                    TIMEOUT 15
            )
        endforeach()
    endforeach()

    foreach(DROP_OLDEST_SCENARIO IN ITEMS schema tensor image)
        string(TOUPPER "${DROP_OLDEST_SCENARIO}" DROP_OLDEST_SCENARIO_UPPER)
        set(DROP_OLDEST_PAYLOAD_ARGUMENTS)
        if(DROP_OLDEST_SCENARIO STREQUAL "tensor" OR DROP_OLDEST_SCENARIO STREQUAL "image")
            list(APPEND DROP_OLDEST_PAYLOAD_ARGUMENTS
                --tensor-storage pageable_host
                --tensor-bytes 64
            )
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
                        --warmup 4
                        --payload-validation all
                        ${DROP_OLDEST_PAYLOAD_ARGUMENTS}
                        ${DROP_OLDEST_DELIVERY_ARGUMENTS}
                        --timeout-ms 10000
            )
            set_tests_properties(
                BENCHMARK_RUNTIME_GRAPH_DROP_OLDEST_${DROP_OLDEST_SCENARIO_UPPER}_${DROP_OLDEST_DELIVERY}
                PROPERTIES
                    LABELS "benchmark;ea1;runtime;lossy;${DROP_OLDEST_SCENARIO}"
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
                LABELS "benchmark;ea1;runtime;capacity"
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
            LABELS "benchmark;ea1;runtime"
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
            LABELS "benchmark;ea1;runtime"
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
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SOURCE_ONLY_NOTIFICATION_LATENCY
        PROPERTIES
            LABELS "benchmark;ea1;runtime"
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
            LABELS "benchmark;ea1;runtime"
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
            LABELS "benchmark;ea1;runtime"
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
                LABELS "benchmark;ea1;runtime"
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
            LABELS "benchmark;ea1;runtime"
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
            LABELS "benchmark;ea1;runtime"
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
            LABELS "benchmark;ea1;runtime;fanout"
            TIMEOUT 15
    )

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
            LABELS "benchmark;ea1;runtime;fanout;placement"
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
                LABELS "benchmark;ea1;runtime;schema"
                TIMEOUT 15
        )
    endforeach()

    # Exercise the generated-schema fanout path independently of the POD payload family.
    add_test(
        NAME BENCHMARK_RUNTIME_GRAPH_SCHEMA_FANOUT
        COMMAND runtime_graph_benchmark
                --workload latency
                --scenario schema
                --fanout 2
                --samples 8
                --warmup 2
                --publisher-period-us 1000
                --payload-validation all
                --timeout-ms 10000
    )
    set_tests_properties(BENCHMARK_RUNTIME_GRAPH_SCHEMA_FANOUT
        PROPERTIES
            LABELS "benchmark;ea1;runtime;fanout;schema"
            TIMEOUT 15
    )

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
                LABELS "benchmark;ea1;runtime;schema;tensor"
                TIMEOUT 15
        )
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
                LABELS "benchmark;ea1;runtime;tensor"
                SKIP_RETURN_CODE 77
                TIMEOUT 15
        )
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
            LABELS "benchmark;ea1;runtime;tensor"
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
                LABELS "benchmark;ea1;runtime;tensor;schema;multi_tensor"
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
            LABELS "benchmark;ea1;runtime;tensor"
            TIMEOUT 15
    )
endif()
